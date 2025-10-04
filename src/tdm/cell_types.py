from collections import defaultdict

"""
Short codes for cell types.
"""

FIBROBLAST = "F"
MACROPHAGE = "M"
TUMOR = "Tu"
ENDOTHELIAL = "En"
T_CELL = "T"
B_CELL = "B"
PLASMA = "Pl"
CD4_T_CELL = "CD4"
CD8_T_CELL = "CD8"

DENDRITIC = "DC"
NK = "NK"
NEUTROPHIL = "Neu"

# Stomach:
GOBLET = "goblet"
PARIETAL = "parietal"
FOVEOLAR = "foveolar"
SMOOTH_MUSCLE = "smooth_muscle"

# fixed order for all analyses
CELL_TYPES_ARRAY = [FIBROBLAST, MACROPHAGE, TUMOR, ENDOTHELIAL, T_CELL, B_CELL, CD4_T_CELL, CD8_T_CELL]

CELL_TYPE_TO_FULL_NAME = {
    FIBROBLAST: "Fibroblast",
    MACROPHAGE: "Macrophage",
    TUMOR: "Tumor",
    ENDOTHELIAL: "Endothelial",
    T_CELL: "T-cell",
    B_CELL: "B-cell",
    CD4_T_CELL: "CD4 T-cell",
    CD8_T_CELL: "CD8 T-cell",
}


def cell_type_to_full_name(cell_type: str) -> str:
    return CELL_TYPE_TO_FULL_NAME.get(cell_type, cell_type)


# fixed order for all analyses
FIBROBLAST_COLOR = "#3B76BC"  # blue
MACROPHAGE_COLOR = "#EC6238"  # red
TUMOR_COLOR = "#96C75A"
ENDOTHELIAL_COLOR = "#F3B15B"
T_CELL_COLOR = "#9D65AB"  # dark purple
B_CELL_COLOR = "#3E754E"  # green
# "#632A7D"

CELL_TYPE_TO_COLOR = defaultdict(
    lambda: "#000000",
    {
        FIBROBLAST: FIBROBLAST_COLOR,
        MACROPHAGE: MACROPHAGE_COLOR,
        TUMOR: TUMOR_COLOR,
        ENDOTHELIAL: ENDOTHELIAL_COLOR,
        T_CELL: T_CELL_COLOR,
        B_CELL: B_CELL_COLOR,
    },
)
CELL_FULL_NAME_TO_COLOR = {CELL_TYPE_TO_FULL_NAME[t]: c for t, c in CELL_TYPE_TO_COLOR.items()}
CELL_TYPE_COLORS_ARRAY = list(CELL_TYPE_TO_COLOR.values())

"""
Frequently used plot axis labels
"""
