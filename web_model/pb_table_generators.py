import re
import os
from .params import param_dict


def derived_params():
    d = param_dict

    # quantity shares (e.g. Q_LA = LA/L)
    Q_LU = 1 - d["Q_LT"] - d["Q_LA"]  # share of land used for recreation
    # Q_EP = 1-Q_EEps-Q_EFi    # share of fossil fuel used for fertilizer prod.

    Q_EEps = 1 - d["Q_EFi"] - d["Q_EP"]  # share of fossil fuel used for fisheries prod.

    Q_AF = 1 - d["Q_AB"]  # share of agri. prod. used for food prod.
    Q_EpsY = 1 - d["Q_EpsA"]  # share of energy used for final goods prod.

    # factor shares (Gamma)
    GammaU_nF = 1 - d["GammaU_F"]  # factor share non-food (utility)
    GammaF_AF = 1 - d["GammaF_Fi"]  # factor share agricultural (food)
    GammanF_T = 1 - d["GammanF_LU"] - d["GammanF_Y"]  # factor share timber (non-food)
    GammaA_nLA = 1 - d["GammaA_LA"]  # factor share non-land (agric. prod)

    GammaEps_R = (
        1 - d["GammaEps_AB"] - d["GammaEps_EEps"]
    )  # factor share renewables (energy services prod.)
    GammaFi_MFi = (
        1 - d["GammaFi_EFi"]
    )  # factor share intermediates (energy services prod.)
    GammaT_MT = 1 - d["GammaT_LT"]  # factor share intermediates (energy services prod.)
    GammaY_MY = (
        1 - d["GammaY_EpsY"]
    )  # factor share intermediates (energy services prod.)
    GammanLA_MA = (
        1 - d["GammanLA_W"] - d["GammanLA_P"] - d["GammanLA_EpsA"]
    )  # factor share intermediates (non-land)
    GammaP_MP = (
        1 - d["GammaP_EP"] - d["GammaP_Pho"]
    )  # factor share intermediates (fertilizer prod.)

    GammaT_LTLT = d["GammaT_LT"] - 1
    GammaY_EpsYEpsY = d["GammaY_EpsY"] - 1

    GammaA_P = GammaA_nLA * d["GammanLA_P"]
    GammaA_W = GammaA_nLA * d["GammanLA_W"]
    GammaA_EpsA = GammaA_nLA * d["GammanLA_EpsA"]
    GammaU_AF = d["GammaU_F"] * GammaF_AF
    GammaU_Fi = d["GammaU_F"] * d["GammaF_Fi"]
    GammaU_Y = d["GammaU_F"] * d["GammanF_Y"]
    GammaU_LU = d["GammaU_F"] * d["GammanF_LU"]
    GammaU_T = d["GammaU_F"] * GammanF_T
    GammaA_MA = GammaA_nLA * GammanLA_MA

    dp = locals()
    del dp["d"]
    return dp


typology = {
    r"\multicolumn{5}{l}{\textit{Agricultural Sector: Production}}": [],
    r"\multicolumn{5}{l}{\textit{Agricultural Sector: Inputs}}": [],
    r"\multicolumn{5}{l}{\textit{Energy-related sectors and services}}": [],
    r"\multicolumn{5}{l}{\textit{Extractive Sectors}}": [],
    r"\multicolumn{5}{l}{\textit{Other}}": [],
}


table_text = {
    "L_A": (
        r"\hspace{0.2cm}Land-Share Agriculture",
        r"\multicolumn{5}{l}{\textit{Agricultural Sector: Inputs}}",
    ),
    "L_T": (r"\hspace{0.2cm}Land-Share Timber", r"\multicolumn{5}{l}{\textit{Other}}",),
    "L_U": (
        r"\hspace{0.2cm}Land-Share Natural",
        r"\multicolumn{5}{l}{\textit{Other}}",
    ),
    "E": (
        r"\hspace{0.2cm}Fossil-Fuel Extraction",
        r"\multicolumn{5}{l}{\textit{Extractive Sectors}}",
    ),
    "E_Eps": (
        r"\hspace{0.2cm}Fossil-Fuel in Energy Services",
        r"\multicolumn{5}{l}{\textit{Energy-related sectors and services}}",
    ),
    "E_P": (
        r"\hspace{0.2cm}Fossil-Fuel in Fertilizer Prod.",
        r"\multicolumn{5}{l}{\textit{Energy-related sectors and services}}",
    ),
    "E_Fi": (
        r"\hspace{0.2cm}Fossil-Fuel in Fisheries",
        r"\multicolumn{5}{l}{\textit{Energy-related sectors and services}}",
    ),
    "A": (
        r"\hspace{0.2cm}Total",
        r"\multicolumn{5}{l}{\textit{Agricultural Sector: Production}}",
    ),
    "A_B": (
        r"\hspace{0.2cm}Biofuels",
        r"\multicolumn{5}{l}{\textit{Agricultural Sector: Production}}",
    ),
    "A_F": (
        r"\hspace{0.2cm}Food",
        r"\multicolumn{5}{l}{\textit{Agricultural Sector: Production}}",
    ),
    "Eps": (
        r"\hspace{0.2cm}Energy Services",
        r"\multicolumn{5}{l}{\textit{Energy-related sectors and services}}",
    ),
    "Eps_A": (
        r"\hspace{0.2cm}Energy in Agriculture",
        r"\multicolumn{5}{l}{\textit{Agricultural Sector: Inputs}}",
    ),
    "Eps_Y": (
        r"\hspace{0.2cm}Energy in Manufacturing",
        r"\multicolumn{5}{l}{\textit{Energy-related sectors and services}}",
    ),
    "P": (
        r"\hspace{0.2cm}Fertilizer Production",
        r"\multicolumn{5}{l}{\textit{Agricultural Sector: Inputs}}",
    ),
    "W": (
        r"\hspace{0.2cm}Water Production",
        r"\multicolumn{5}{l}{\textit{Agricultural Sector: Inputs}}",
    ),
    "Pho": (
        r"\hspace{0.2cm}Phosphate Extraction",
        r"\multicolumn{5}{l}{\textit{Extractive Sectors}}",
    ),
    "R": (
        r"\hspace{0.2cm}Renewables Production",
        r"\multicolumn{5}{l}{\textit{Energy-related sectors and services}}",
    ),
    "Fi": (
        r"\hspace{0.2cm}Fisheries Production",
        r"\multicolumn{5}{l}{\textit{Other}}",
    ),
    "T": (r"\hspace{0.2cm}Timber Production", r"\multicolumn{5}{l}{\textit{Other}}"),
    "Y": (r"\hspace{0.2cm}Manufacturing", r"\multicolumn{5}{l}{\textit{Other}}"),
}


def model_variable_sensitivity_table(var_desc_dict, df_carbontax, df_carbonbiofueltax):

    for key, var_name in var_desc_dict.items():
        if key in table_text.keys():

            quant_index_series = df_carbontax["description"].str.contains(
                var_name[1], regex=False
            )
            quant_index = quant_index_series.index[quant_index_series].tolist()[0]

            # print(quant_index, key, price_name, price_index, price_desc_dict[price_name][1])
            latex_str = table_text[key][
                0
            ]  # df_full.iloc[quant_index, df_full.columns.get_loc('index')]

            minc = df_carbontax.iloc[
                quant_index, df_carbontax.columns.get_loc("min value")
            ]
            maxc = df_carbontax.iloc[
                quant_index, df_carbontax.columns.get_loc("max value")
            ]
            mincb = df_carbonbiofueltax.iloc[
                quant_index, df_carbonbiofueltax.columns.get_loc("min value")
            ]
            maxcb = df_carbonbiofueltax.iloc[
                quant_index, df_carbonbiofueltax.columns.get_loc("max value")
            ]

            latex_str += " & " + str(round(minc, 4))
            latex_str += " & " + str(round(maxc, 4))
            latex_str += " & " + str(round(mincb, 4))
            latex_str += " & " + str(round(maxcb, 4))
            latex_str += " \\\\ "

            typology[table_text[key][1]].append(latex_str)

    print(r"\begin{tabular}{lrrrr}")
    print(r"\hline")
    print(r"& \multicolumn{2}{c}{}  & \multicolumn{2}{c}{\textit{Carbon Tax +}}  \\ ")
    print(
        r"& \multicolumn{2}{c}{\textit{Carbon tax}}  & \multicolumn{2}{c}{\textit{Biofuel policy} }  \\ "
    )
    print(
        r"\textit{Variable} & \textit{Min} & \textit{Max} & \textit{Min} & \textit{Max}  \\ "
    )
    print(r"\hline")
    for k, values in typology.items():
        print(k + " \\\\ ")
        for v in values:
            print(v)
    print(r"\hline")
    print(r"\end{tabular}")


def model_variable_result_table(var_desc_dict, price_desc_dict, df_full, df_price_full):
    latex_columns = [
        "outcome",
        "prices",
        "outcome_biofuel",
        "prices_biofuel",
    ]  # , 'outcome_land',  'prices_land'

    for key, var_name in var_desc_dict.items():
        if key in table_text.keys():
            price_name = "p_" + key.split("_")[0]
            price_index_series = df_price_full["index"].str.contains(
                price_desc_dict[price_name][1], regex=False
            )
            price_index = price_index_series.index[price_index_series].tolist()[0]
            quant_index_series = df_full["index"].str.contains(var_name[1], regex=False)
            quant_index = quant_index_series.index[quant_index_series].tolist()[0]

            # print(quant_index, key, price_name, price_index, price_desc_dict[price_name][1])
            latex_str = table_text[key][
                0
            ]  # df_full.iloc[quant_index, df_full.columns.get_loc('index')]

            for col_name in latex_columns:
                if "outcome" in col_name:
                    value = df_full.iloc[quant_index, df_full.columns.get_loc(col_name)]
                else:
                    value = df_price_full.iloc[
                        price_index, df_price_full.columns.get_loc(col_name)
                    ]

                latex_str += " & " + str(round(value, 3))
            latex_str += " \\\\ "
            # latex_str = re.escape(latex_str)

            typology[table_text[key][1]].append(latex_str)

    print(r"\begin{tabular}{lrrrr}")
    print(r"\hline")
    print(r"& \multicolumn{2}{c}{}  & \multicolumn{2}{c}{\textit{Carbon Tax +}}  \\ ")
    print(
        r"& \multicolumn{2}{c}{\textit{Carbon tax}}  & \multicolumn{2}{c}{\textit{Biofuel policy} }  \\ "
    )
    print(
        r"\textit{Variable} & \textit{Quantity} & \textit{Price} & \textit{Quantity} & \textit{Price}  \\ "
    )
    print(r"\hline")
    for k, values in typology.items():
        print(k + " \\\\ ")
        for v in values:
            print(v)
    print(r"\hline")
    print(r"\end{tabular}")
    return typology


def model_pbimpact_result_table(df_final_impact):
    # Latex code
    def p(x):
        return " ($" + x + "$) "

    def r(x):
        return str(round(x, 4))

    header = r"  & \textit{Climate} & \textit{Biodiv.} & \textit{Biochem.} & \textit{Freshw.} & \textit{Ocean acid.} & \textit{Land-use}  & \textit{Aerosols} & \textit{Ozone} & \textit{Chem.} \\ "
    # header = ' & \\textit{GtCO$_2$ yr-1} & \\textit{threat. species yr-1} & \\textit{Tg N yr–1; Tg P yr–1} & \\textit{km3 yr–1} & \\textit{GtCO$_2$ yr-1} & \\textit{\% of forest}  & \\textit{\% of AOD} & \\textit{(+/-) of Ozone} & \\textit{(+/-) of Chem.} \\\\'

    print(r"\begin{tabular}{llllllllll} ")
    print(header)
    print(r"\hline")
    for _, variable in df_final_impact.iterrows():
        row = ""
        for i, col in enumerate(variable):
            if i == 0:
                row = col
            elif i == 1:
                row += p(col)
            else:
                row += " & " + r(col)
        print(row + r" \\ ")
    tot = df_final_impact.sum()
    chem_tot = -1 * all(df_final_impact.loc[:, "Chem. effect"].values <= 0)
    if chem_tot > -1:
        chem_tot = 1 * all(df_final_impact.loc[:, "Chem. effect"].values >= 0)
    ozone_tot = -1 * all(df_final_impact.loc[:, "Ozone effect"].values <= 0)
    if ozone_tot > -1:
        ozone_tot = 1 * all(df_final_impact.loc[:, "Ozone effect"].values >= 0)
    print(r"\hline")
    print(
        r"Total impact & {} & {} & {} & {} & {} & {} & {} & {} & {} \\ ".format(
            r(tot[2]),
            r(tot[3]),
            r(tot[4]),
            r(tot[5]),
            r(tot[6]),
            r(tot[7]),
            r(tot[8]),
            chem_tot,
            ozone_tot,
        )
    )
    print(r"\hline ")
    print(r"\end{tabular} ")


def param_elast_table():
    with open("./web_model/params_elast_table.txt", "r") as f:
        pars = {**param_dict}
        for line in f.readlines():
            pline = line.replace("\n", "")
            for k, v in pars.items():
                if isinstance(v, list):
                    pline = pline.replace(" " + k, " " + str(v))
                else:
                    val = round(v, 2)
                    pline = pline.replace(" " + k, " " + str(val))

            print(pline)


def param_qshare_table():
    with open("./web_model/params_qshare_table.txt", "r") as f:
        pars = {**param_dict, **derived_params()}
        for line in f.readlines():
            pline = line.replace("\n", "")
            for k, v in pars.items():
                if not isinstance(v, list):
                    val = round(v * 100, 1)
                    pline = pline.replace(k, str(val) + "\%")

            print(pline)


def param_expshare_table():
    print(len(derived_params()))
    with open("./web_model/params_expshare_table.txt", "r") as f:
        pars = {**param_dict, **derived_params()}
        for line in f.readlines():
            pline = line.replace("\n", "")
            for k, v in pars.items():
                if not isinstance(v, list):
                    val = round(v * 100, 1)
                    pline = pline.replace(k, str(val) + "\%")
            print(pline)

    # # quantity shares (e.g. Q_LA = LA/L)
    # Q_LU = 1 - d["Q_LT"] - d["Q_LA"]  # share of land used for recreation
    # # Q_EP = 1-Q_EEps-Q_EFi    # share of fossil fuel used for fertilizer prod.

    # Q_EEps = 1 - d["Q_EFi"] - d["Q_EP"]  # share of fossil fuel used for fisheries prod.

    # Q_AF = 1 - d["Q_AB"]  # share of agri. prod. used for food prod.
    # Q_EpsY = 1 - d["Q_EpsA"]  # share of energy used for final goods prod.
