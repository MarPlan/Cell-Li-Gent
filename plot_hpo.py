import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    ForbiddenAndConjunction,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    Integer,
    Normal,
)
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skopt import Space
from skopt.plots import plot_objective
from skopt.utils import create_result
from smac.runhistory.runhistory import RunHistory

# mpl.rcParams["font.family"] = "serif"
# mpl.rcParams["font.serif"] = ["Computer Modern Serif"]
# mpl.rcParams["font.size"] = 9  # Set the font size to 11
# mpl.rcParams["text.usetex"] = True  # Use LaTeX to render the text


def plot_hpo_partial(save=False):
    cs = ConfigurationSpace(
        name="transformer",
        space={
            "pe_type": Categorical("pe_type", ["APE", "RoPE", "ALiBi"]),
            "norm_type": Categorical("norm_type", ["RMSNorm", "LayerNorm"]),
            "rope_theta": Float("rope_theta", bounds=(500, 200_000)),
            # "loss": Categorical("loss", ["MSE", "MAE"]),
            # "reduction": Categorical("reduction", ["sum", "mean"]),
            "dim_model": Categorical(
                "dim_model", [64, 128, 256, 384, 512, 768], ordered=True
            ),
            "n_heads": Categorical(
                "n_heads",
                [2, 4, 8, 12, 16, 32, 64],
                ordered=True,
            ),
            "seq_len": Categorical("seq_len", [128, 256, 512, 768, 1024, 1536], ordered=True),
            "n_layer": Integer("n_layer", bounds=(8, 25)),
            # "bias": Categorical("bias", [True, False], default=False),
            # "learning_rate": Float(
            #    "learning_rate",
            #    bounds=(1e-5, 1e-2),
            #    # log=True,
            #    default=1.5e-3,
            #    distribution=Normal(mu=5e-3, sigma=2),
            # ),
        },
    )
 
    cs.add_condition(EqualsCondition(cs["rope_theta"], cs["pe_type"], "RoPE"))
 
    # Function to find the forbidden heads for a given dim_model.
    def forbidden_heads_for_dim_model(dim_model, n_heads):
        return [head for head in n_heads if head >= dim_model or dim_model % head != 0]
 
    # Creating all forbidden clauses.
    forbidden_clauses = []
    for dim_model in cs["dim_model"].sequence:
        forbidden_heads = forbidden_heads_for_dim_model(
            dim_model, cs["n_heads"].sequence
        )
        if forbidden_heads:
            forbidden_dim_clause = ForbiddenEqualsClause(cs["dim_model"], dim_model)
            forbidden_heads_clause = ForbiddenInClause(cs["n_heads"], forbidden_heads)
            forbidden_clauses.append(
                ForbiddenAndConjunction(forbidden_dim_clause, forbidden_heads_clause)
            )
 
    cs.add_forbidden_clauses(forbidden_clauses)
 
    forbidden_dim_flash_attn = ForbiddenEqualsClause(cs["dim_model"], 768)
    forbidden_head_flash_attn = ForbiddenEqualsClause(cs["n_heads"], 2)
    forbidden_flash_attn = ForbiddenAndConjunction(
        forbidden_dim_flash_attn, forbidden_head_flash_attn
    )
    cs.add_forbidden_clauses([forbidden_flash_attn])

    # Create a RunHistory object
    runhistory = RunHistory()

    # Load the run history from a JSON file
    runhistory.load("hpo/transformer_final/0/runhistory.json", cs)

    # Extract the configuration IDs and corresponding costs
    extracted_data = {"budget": [], "config_id": [], "cost": [], "starttime": [], "endtime": []}

    for trial_key, trial_value in runhistory._data.items():
        extracted_data["budget"].append(trial_key.budget)
        extracted_data["config_id"].append(trial_key.config_id)
        extracted_data["cost"].append(trial_value.cost)
        extracted_data["starttime"].append(trial_value.starttime)
        extracted_data["endtime"].append(trial_value.endtime)

    # Extract the hyperparameters for each configuration
    hyperparameters = []
    hyperparameters = [
        dict(runhistory.get_config(config_id))
        for config_id in extracted_data["config_id"]
    ]

    # Define the mappings
    # activation_function_map = {"GeLU": 0, "SwiGLU": 1}
    # loss_function_map = {"MAE": 0, "MSE": 1}
    normalization_map = {"RMSNorm": 0, "LayerNorm": 1}
    positional_encoding_map = {"ALiBi": 0,"RoPE": 1, "APE":2}
    reduction_map = {"sum": 0, "mean": 1}

    # Encode the data
    for item in hyperparameters:
    # item['act_type'] = activation_function_map[item['act_type']]
    # item['loss'] = loss_function_map[item['loss']]
        item['norm_type'] = normalization_map[item['norm_type']]
        item['pe_type'] = positional_encoding_map[item['pe_type']]
        # item['reduction'] = reduction_map[item['reduction']]

    # Convert the hyperparameters to a format suitable for the Result object
    hypparam_names = list(cs.keys())
    hyperparameters = [
        [
            config[name] if name != "rope_theta" else (config.get(name, 66_666))
            for name in hypparam_names
        ]
        for config in hyperparameters
    ]

    # Assume `hyperparameters` is a list of configurations and `costs` is a list of corresponding costs
    hyperparameters = np.array(hyperparameters).astype(np.float64)
    # hyperparameters= np.concatenate((hyperparameters, np.array(extracted_data["budget"]).reshape(-1, 1)), axis=1)
    # hypparam_names.append("budget")

    costs = np.array(extracted_data["cost"]).astype(np.float64)
    # mask = costs != 1e7
    mask = (costs != 1e7) & (np.array(extracted_data["budget"]) > 3200)
    costs = costs[mask]
    hyperparameters = hyperparameters[mask]

    # Remove rope_theta
    # hyperparameters = hyperparameters[:,:-1]
    # hypparam_names = hypparam_names[:-1]

    space = Space([(hp.min(), hp.max()) for hp in hyperparameters.T])

    # Define standard scalers for X and y separately
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Scale X and y
    X_scaled = scaler_X.fit_transform(hyperparameters)
    y_scaled = scaler_y.fit_transform(costs.reshape(-1, 1)).flatten()

    # Define your SVR model
    svr = SVR()

    # Train the SVR model on the scaled data
    svr.fit(X_scaled, y_scaled)

    class CustomSVR:
        def __init__(self, svr_model, scaler_X, scaler_y):
            self.svr_model = svr_model
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y

        def predict(self, X):
            # Scale the input features
            X_scaled = self.scaler_X.transform(X)

            # Predict in the scaled space
            y_pred_scaled = self.svr_model.predict(X_scaled)

            # Unscale the predictions
            y_pred_unscaled = self.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()

            return y_pred_unscaled

    # Create Custom SVR instance
    model = CustomSVR(svr, scaler_X, scaler_y)

    # Create a Scikit-Optimize Result object
    result = create_result(
        Xi=hyperparameters.tolist(), yi=costs.tolist(), space=space, models=[model]
    )
    # Now you can use the Result object with the plot_objective function
    # _ = plot_evaluations(result)
    ax = plot_objective(
        result,
        n_points=120,
        sample_source="result",
        levels=6,
        # cmap=mpl.colormaps["YlOrRd"],
        cmap="YlOrRd",
        dimensions=hypparam_names,
        # sample_source="expected_minimum",
        # n_points=50,
        # n_minimum_search=400,
    )
    # Set the figure size
    fig = plt.gcf()  # Get the current figure
    fig_size_big = (15, 15)
    fig.set_size_inches(fig_size_big)
    # # Tmax
    # ax[0, 0].set_xlim(5, 15)
    # ax[0, 0].set_xlabel("")
    # ax[0, 0].set_xticks([6, 10, 14])
    # ax[0, 0].set_xticklabels([])
    # ax[0, 0].set_ylabel("")
    #
    # ax[1, 0].set_xlim(5, 15)
    # ax[1, 0].set_xticks([6, 10, 14])
    #
    # ax[2, 0].set_xlim(5, 15)
    # ax[2, 0].set_xticks([6, 10, 14])
    #
    # ax[3, 0].set_xlim(5, 15)
    # ax[3, 0].set_xticks([6, 10, 14])
    #
    # ax[4, 0].set_xlim(5, 15)
    # ax[4, 0].set_xticks([6, 10, 14])
    #
    # ax[5, 0].set_xlim(5, 15)
    # ax[5, 0].set_xticks([6, 10, 14])
    # ax[5, 0].xaxis.labelpad = 14  # Set padding to 20, adjust as needed
    #
    # # Dropout
    # ax[1, 0].set_ylim(0.05, 0.3)
    # ax[1, 0].set_yticks([0.1, 0.175, 0.25])
    # # ax[1, 0].yaxis.labelpad = 20  # Set padding to 20, adjust as needed
    #
    # ax[1, 1].set_xlim(0.05, 0.3)
    # ax[1, 1].set_xlabel("")
    # ax[1, 1].set_xticks([0.1, 0.175, 0.25])
    # ax[1, 1].set_xticklabels([])
    # ax[1, 1].set_ylabel("")
    #
    # ax[2, 1].set_xlim(0.05, 0.3)
    # ax[2, 1].set_xticks([0.1, 0.175, 0.25])
    #
    # ax[3, 1].set_xlim(0.05, 0.3)
    # ax[3, 1].set_xticks([0.1, 0.175, 0.25])
    #
    # ax[4, 1].set_xlim(0.05, 0.3)
    # ax[4, 1].set_xticks([0.1, 0.175, 0.25])
    #
    # ax[5, 1].set_xlim(0.05, 0.3)
    # ax[5, 1].set_xticks([0.1, 0.175, 0.25])
    # # ax[5, 0].xaxis.labelpad = 20  # Set padding to 20, adjust as needed
    #
    # # Layers
    # ax[2, 0].set_ylim(2, 14)
    # ax[2, 0].set_yticks([4, 8, 12])
    # ax[2, 0].yaxis.labelpad = 18  # Set padding to 20, adjust as needed
    #
    # ax[2, 1].set_ylim(2, 14)
    # ax[2, 1].set_yticks([4, 8, 12])
    #
    # ax[2, 2].set_xlim(2, 14)
    # ax[2, 2].set_xlabel("")
    # ax[2, 2].set_xticks([4, 8, 12])
    # ax[2, 2].set_xticklabels([])
    # ax[2, 2].set_ylabel("")
    #
    # ax[3, 2].set_xlim(2, 14)
    # ax[3, 2].set_xticks([4, 8, 12])
    #
    # ax[4, 2].set_xlim(2, 14)
    # ax[4, 2].set_xticks([4, 8, 12])
    #
    # ax[5, 2].set_xlim(2, 14)
    # ax[5, 2].set_xticks([4, 8, 12])
    # ax[5, 2].xaxis.labelpad = 14  # Set padding to 20, adjust as needed
    #
    # # Lr
    # ax[3, 0].set_ylim(0, 0.035)
    # ax[3, 0].set_yticks([0.005, 0.0175, 0.03])
    # ax[3, 0].set_yticklabels([0.005, 0.018, 0.03])
    #
    # ax[3, 1].set_ylim(0, 0.035)
    # ax[3, 1].set_yticks([0.005, 0.0175, 0.030])
    #
    # ax[3, 2].set_ylim(0, 0.035)
    # ax[3, 2].set_yticks([0.005, 0.0175, 0.030])
    #
    # ax[3, 3].set_xlim(0, 0.035)
    # ax[3, 3].set_xlabel("")
    # ax[3, 3].set_xticks([0.005, 0.0175, 0.030])
    # ax[3, 3].set_xticklabels([])
    # ax[3, 3].set_ylabel("")
    #
    # ax[4, 3].set_xlim(0, 0.035)
    # ax[4, 3].set_xticks([0.005, 0.0175, 0.030])
    #
    # ax[5, 3].set_xlim(0, 0.035)
    # ax[5, 3].set_xticks([0.005, 0.0175, 0.030])
    # ax[5, 3].set_xticklabels([0.005, 0.018, 0.030])
    #
    # # Seqlen
    # ax[4, 0].set_ylim(150, 400)
    # ax[4, 0].set_yticks([200, 350])
    # ax[4, 0].yaxis.labelpad = 12  # Set padding to 20, adjust as needed
    #
    # ax[4, 1].set_ylim(150, 400)
    # ax[4, 1].set_yticks([200, 350])
    #
    # ax[4, 2].set_ylim(150, 400)
    # ax[4, 2].set_yticks([200, 350])
    #
    # ax[4, 3].set_ylim(150, 400)
    # ax[4, 3].set_yticks([200, 350])
    #
    # ax[4, 4].set_xlim(150, 400)
    # ax[4, 4].set_xlabel("")
    # ax[4, 4].set_xticks([200, 350])
    # ax[4, 4].set_xticklabels([])
    # ax[4, 4].set_ylabel("")
    #
    # ax[5, 4].set_xlim(150, 400)
    # ax[5, 4].set_xticks([200, 350])
    # ax[5, 4].xaxis.labelpad = 10  # Set padding to 20, adjust as needed
    #
    # # Weight decay
    # ax[5, 0].set_ylim(0.001, 0.1)
    # ax[5, 0].set_yticks([0.02, 0.05, 0.08])
    # ax[5, 0].yaxis.labelpad = 9  # Set padding to 20, adjust as needed
    #
    # ax[5, 1].set_ylim(0.001, 0.1)
    # ax[5, 1].set_yticks([0.02, 0.05, 0.08])
    #
    # ax[5, 2].set_ylim(0.001, 0.1)
    # ax[5, 2].set_yticks([0.02, 0.05, 0.08])
    #
    # ax[5, 3].set_ylim(0.001, 0.1)
    # ax[5, 3].set_yticks([0.02, 0.05, 0.08])
    #
    # ax[5, 4].set_ylim(0.001, 0.1)
    # ax[5, 4].set_yticks([0.02, 0.05, 0.08])
    #
    # ax[5, 5].set_xlim(0.001, 0.1)
    # ax[5, 5].set_xlabel("")
    # ax[5, 5].set_xticks([0.02, 0.05, 0.08])
    # ax[5, 5].set_xticklabels([])
    # ax[5, 5].set_ylabel("")

    cbar_ax_3d = fig.add_axes([0.45, 0.9, 0.5, 0.03])  # [left, bottom, width, height]
    norm = Normalize(vmin=0, vmax=1)
    # Add colorbars for both colormaps
    sm = plt.cm.ScalarMappable(cmap=mpl.colormaps["YlOrRd"], norm=norm)
    cbar_3d = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.5,
        aspect=15,
        pad=-0.1,
        cax=cbar_ax_3d,
        orientation="horizontal",
    )

    cbar_3d.set_label("Loss", labelpad=-10)
    cbar_3d.set_ticks([0, 1])
    cbar_3d.set_ticklabels(["Low", "High"])
    # if save:
    # # Save the plot
    plt.savefig(
        "hpo_partial.png",
        format="png",
        bbox_inches="tight",
        # pad_inches=[0, 0, 1, 0]
        # pad_inches="tight"
        dpi=300,
    )

    plt.show()


if __name__ == "__main__":
    plot_hpo_partial()
