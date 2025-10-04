import seaborn as sns
from matplotlib.markers import MarkerStyle


def set_style():
    sns.set_style("ticks")
    sns.set_palette(["#3B76BC", "#3E754E", "#96C75A", "#F3B15B", "#E28A71", "#EC6238", "#632A7D"])
    # plt.rcParams["axes.facecolor"] = "#ECECED"


RESPONSE_COLOR = "#3F3C3C"  # black
NO_RESPONSE_COLOR = "#CA493C"  # red

DIVISION_COLOR = "#3B76BC"
DEATH_COLOR = "#EC6238"

# TODO remove, old:
STABLE_POINT_COLOR = "#009780"
UNSTABLE_POINT_COLOR = "#F1BF12"

STABLE = "stable"
SEMI_STABLE = "semi-stable"
UNSTABLE = "unstable"

STABILITIY_TYPE_TO_COLOR = {
    STABLE: "#3E754E",
    SEMI_STABLE: "#FFFFFF",
    UNSTABLE: "#E62525",
}

STABILITIY_TYPE_TO_MARKER_STYLE_KWARGS = {
    STABLE: {
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "marker": MarkerStyle("o"),
        "markersize": 10,
    },
    SEMI_STABLE: {
        "markerfacecolor": "black",
        "markerfacecoloralt": "white",
        "markeredgecolor": "black",
        "marker": MarkerStyle("o", fillstyle="top"),
        "markersize": 10,
    },
    UNSTABLE: {
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "marker": MarkerStyle("o"),
        "markersize": 10,
    },
}
