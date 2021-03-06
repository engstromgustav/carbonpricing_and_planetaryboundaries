import pandas as pd
from collections import OrderedDict

param_dict = {
    "tau_E": 0.0,
    "V_A": 0.05,
    "V_T": 0.05,
    "GammaP_Pho": 0.3127,
    "GammanLA_EpsA": 0.0412,
    "GammaY_EpsY": 0.0638,
    "GammaT_LT": 0.3748,
    "GammaFi_EFi": 0.228,
    "GammaEps_EEps": 0.9433,
    "GammaEps_AB": 0.0037,
    "GammaP_EP": 0.1095,
    "GammanLA_W": 0.0239,
    "GammanLA_P": 0.0796,
    "GammaA_LA": 0.192,
    "GammanF_LU": 0.01711,
    "GammanF_Y": 0.991,
    "GammaF_Fi": 0.034,
    "GammaU_F": 0.1235,
    "Q_EFi": 0.004,
    "Q_EP": 0.014,
    "Q_EpsA": 0.05,
    "Q_AB": 0.038,
    "Q_LT": 0.02,
    "Q_LA": 0.53,
    "Lambda_MFi": [0, 2, 1],
    "Lambda_MP": [0, 2, 1],
    "Lambda_MY": [0, 2, 1],
    "Lambda_MT": [0, 2, 1],
    "Lambda_MA": [0, 2, 1],
    "Lambda_M": [0, 2, 1],
    "Lambda_Pho": 1/1.5,
    "Lambda_W": 1/1.79,
    "Lambda_E": [0.8, 1.2, 1],
    "Lambda_R": 1/2.7,
    "sigma_Y": [0.1, 1, 0.5],
    "sigma_T": [0.1, 1, 0.2],
    "sigma_Fi": [0.1, 1, 0.2],
    "sigma_Eps": [1.5, 2.1, 1.8],
    "sigma_nLA": [0.25, 0.75, 0.5],
    "sigma_P": [0.05, 0.3, 0.2],
    "sigma_A": [1.10, 1.24, 1.14],
    "sigma_nF": [1.5, 2.1, 1.8],
    "sigma_F": [1.13, 1.33, 1.23],
    "sigma_U": [0.4, 0.6, 0.5],
}

df_typing_formatting = pd.DataFrame(
    OrderedDict(
        [
            (
                "type",
                [
                    "Elasticity of substitution",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "Supply elasticities",
                    "",
                    "",
                    "",
                    "",
                    "Quantity shares",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "Factor shares",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "Elasticity of conversion costs",
                    "",
                    "Carbon tax",
                ],
            ),
            (
                "name",
                [
                    "Food and non food (utility) ",
                    "Agric. output and fish in food (nested utility)",
                    "Timber, manufacturing and land products (nested utility)",
                    "Land and non-land (agric. prod)",
                    "Phosphor and fossil fuels (fertil. prod)",
                    "Phosphor, energy and water (nested agric. prod)",
                    "Fossil fuel, renewables and biofuel (energy prod)",
                    "Fossil fuels, intermediate inputs",
                    "Land, intermediate inputs",
                    "Energy services, intermediate inputs",
                    "Renewables",
                    "Fossil fuel",
                    "Water",
                    "Phosphate",
                    "Intermediates",
                    "Share of land used for agriculture",
                    "Share of land used for timber",
                    "Share of agri. prod. used for biofuels prod.",
                    "Share of energy used for agri. prod.",
                    "Share of fossil fuel used for fertilizer prod.",
                    "Share of fossil fuel used for energy prod.",
                    "Food (utility)",
                    "Fish (food)",
                    "Manufacturing (non-food)",
                    "Unused land (non-food)",
                    "Land (agric. prod)",
                    "Phospor (non-land)",
                    "Water (non-land)",
                    "Fossil fuel (fertilizer prod.)",
                    "Biofuels (energy services prod.)",
                    "Fossil fuel (energy services prod.)",
                    "Fossil fuel (fisheries.)",
                    "Land (timber prod.)",
                    "Energy (manufacturing.)",
                    "Energy services in (non-land)",
                    "Phosphate (fertilizer prod.)",
                    "Conversion costs timber",
                    "Conversion costs agriculture",
                    "Carbon tax",
                ],
            ),
            (
                "values",
                [
                    param_dict["sigma_U"],
                    param_dict["sigma_F"],
                    param_dict["sigma_nF"],
                    param_dict["sigma_A"],
                    param_dict["sigma_P"],
                    param_dict["sigma_nLA"],
                    param_dict["sigma_Eps"],
                    param_dict["sigma_Fi"],
                    param_dict["sigma_T"],
                    param_dict["sigma_Y"],
                    param_dict["Lambda_R"],
                    param_dict["Lambda_E"],
                    param_dict["Lambda_W"],
                    param_dict["Lambda_Pho"],
                    param_dict["Lambda_M"],
                    param_dict["Q_LA"],
                    param_dict["Q_LT"],
                    param_dict["Q_AB"],
                    param_dict["Q_EpsA"],
                    param_dict["Q_EP"],
                    param_dict["Q_EFi"],
                    param_dict["GammaU_F"],
                    param_dict["GammaF_Fi"],
                    param_dict["GammanF_Y"],
                    param_dict["GammanF_LU"],
                    param_dict["GammaA_LA"],
                    param_dict["GammanLA_P"],
                    param_dict["GammanLA_W"],
                    param_dict["GammaP_EP"],
                    param_dict["GammaEps_AB"],
                    param_dict["GammaEps_EEps"],
                    param_dict["GammaFi_EFi"],
                    param_dict["GammaT_LT"],
                    param_dict["GammaY_EpsY"],
                    param_dict["GammanLA_EpsA"],
                    param_dict["GammaP_Pho"],
                    param_dict["V_T"],
                    param_dict["V_A"],
                    param_dict["tau_E"],
                ],
            ),
            (
                "keys",
                [
                    "sigma_U",
                    "sigma_F",
                    "sigma_nF",
                    "sigma_A",
                    "sigma_P",
                    "sigma_nLA",
                    "sigma_Eps",
                    "sigma_Fi",
                    "sigma_T",
                    "sigma_Y",
                    "Lambda_R",
                    "Lambda_E",
                    "Lambda_W",
                    "Lambda_Pho",
                    "Lambda_M",
                    "Q_LA",
                    "Q_LT",
                    "Q_AB",
                    "Q_EpsA",
                    "Q_EP",
                    "Q_EFi",
                    "GammaU_F",
                    "GammaF_Fi",
                    "GammanF_Y",
                    "GammanF_LU",
                    "GammaA_LA",
                    "GammanLA_P",
                    "GammanLA_W",
                    "GammaP_EP",
                    "GammaEps_AB",
                    "GammaEps_EEps",
                    "GammaFi_EFi",
                    "GammaT_LT",
                    "GammaY_EpsY",
                    "GammanLA_EpsA",
                    "GammaP_Pho",
                    "V_T",
                    "V_A",
                    "tau_E",
                ],
            ),
        ]
    )
)


var_desc_dict = {
    "L_A": ("$L_A$", "(land-share agriculture)"),
    "L_T": ("$L_T$", "(land-share timber)"),
    "L_U": ("$L_U$", "(land-share other)"),
    "E": ("$E$", "(fossil-fuel extracted)"),
    "E_Eps": ("$E_{\mathcal{E}}$", "(fossil-fuel use energy serv.)"),
    "E_P": ("$E_P$", "(fossil-fuel use fertilizer prod.)"),
    "E_Fi": ("$E_F$", "(fossil-fuel use fisheries)"),
    "A": ("$A\;\;$", "(agriculture total production)"),
    "A_B": ("$A_B$", "(agriculture prod. for biofuels)"),
    "A_F": ("$A_F$", "(agriculture prod. for food)"),
    "Eps": ("$\mathcal{E}\;\;$", "(energy services)"),
    "Eps_A": ("$\mathcal{E}_A$", "(energy for agriculture)"),
    "Eps_Y": ("$\mathcal{E}_{Y}$", "(energy services for final goods)"),
    "P": ("$P\;\;$", "(fertilizer production)"),
    "W": ("$W\;$", "(water production)"),
    "Pho": ("$\mathcal{P}\;\;$", "(phosphate extraction)"),
    "R": ("$R\;\;$", "(renewables production)"),
    "Fi": ("$F\;\;$", "(fisheries production)"),
    "T": ("$T\;\;$", "(timber production)"),
    "Y": ("$Y\;\;$", "(final goods)"),
    "MA": ("$M_A\;\;$", "(intermediaries agriculture.)"),
    "MT": ("$M_T\;\;$", "(intermediaries timber.)"),
    "MFi": ("$M_F\;\;$", "(intermediaries fisheries)"),
    "MY": ("$M_Y\;\;$", "(intermediaries final goods.)"),
    "MP": ("$M_P\;\;$", "(intermediaries fertilizers)"),
}

# prepare list of price changes and labels
price_desc_dict = {
    "p_A": ("$p_A$", "(agricultural goods)"),
    "p_E": ("$p_E$", "(fossil-fuels)"),
    "p_Eps": ("$p_{\mathcal{E}}$", "(energy services)"),
    "p_Fi": ("$p_{F}$", "(fisheries)"),
    "p_L": ("$p_{L}$", "(land)"),
    "p_MA": ("$p_{M_A}$", "(intermediaries agric.)"),
    "p_MFi": ("$p_{M_F}$", "(intermediaries fisheries)"),
    "p_MP": ("$p_{M_P}$", "(intermediaries fertilizers.)"),
    "p_MT": ("$p_{M_T}$", "(intermediaries timber)"),
    "p_MY": ("$p_{M_Y}$", "(intermediaries final goods)"),
    "p_P": ("$p_{P}$", "(fertilizers)"),
    "p_Pho": ("$p_{Pho}$", "(phosphate)"),
    "p_R": ("$p_{R}$", "(renewables)"),
    "p_T": ("$p_{T}$", "(timber)"),
    "p_W": ("$p_{W}$", "(water)"),
    "p_Y": ("$p_{Y}$", "(final goods)"),
}
