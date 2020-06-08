import re



typology = {
    "\multicolumn{5}{l}{\\textit{Agricultural Sector: Production}}": [],
    "\multicolumn{5}{l}{\\textit{Agricultural Sector: Inputs}}": [],
    "\multicolumn{5}{l}{\\textit{Energy-related sectors and  Services}}": [],
    "\multicolumn{5}{l}{\\textit{Extractive Sectors}}": [],
    "\multicolumn{5}{l}{\\textit{Others}}": [],
}



table_text = {
    "L_A": (
        "\hspace{0.2cm}Land-Share Agriculture",
        "\multicolumn{5}{l}{\\textit{Agricultural Sector: Inputs}}",
    ),
    "L_T": (
        "\hspace{0.2cm}Land-Share Timber",
        "\multicolumn{5}{l}{\\textit{Others}}",
    ),
    "L_U": (
        "\hspace{0.2cm}Land-Share Natural",
        "\multicolumn{5}{l}{\\textit{Others}}",
    ),
    "E": (
        "\hspace{0.2cm}Fossil-Fuel Extraction",
        "\multicolumn{5}{l}{\\textit{Extractive Sectors}}",
    ),
    "E_Eps": (
        "\hspace{0.2cm}Fossil-Fuel in Energy Services",
        "\multicolumn{5}{l}{\\textit{Energy-related sectors and  Services}}",
    ),
    "E_P": (
        "\hspace{0.2cm}Fossil-Fuel in Fertilizer Prod.",
        "\multicolumn{5}{l}{\\textit{Energy-related sectors and  Services}}",
    ),
    "E_Fi": (
        "\hspace{0.2cm}Fossil-Fuel in Fisheries",
        "\multicolumn{5}{l}{\\textit{Energy-related sectors and  Services}}",
    ),
    "A": (
        "\hspace{0.2cm}Total",
        "\multicolumn{5}{l}{\\textit{Agricultural Sector: Production}}",
    ),
    "A_B": (
        "\hspace{0.2cm}Biofuels",
        "\multicolumn{5}{l}{\\textit{Agricultural Sector: Production}}",
    ),
    "A_F": (
        "\hspace{0.2cm}Food",
        "\multicolumn{5}{l}{\\textit{Agricultural Sector: Production}}",
    ),
    "Eps": (
        "\hspace{0.2cm}Energy Services",
        "\multicolumn{5}{l}{\\textit{Energy-related sectors and  Services}}",
    ),
    "Eps_A": (
        "\hspace{0.2cm}Energy in Agriculture",
        "\multicolumn{5}{l}{\\textit{Agricultural Sector: Inputs}}",
    ),
    "Eps_Y": (
        "\hspace{0.2cm}Energy in Manufacturing",
        "\multicolumn{5}{l}{\\textit{Energy-related sectors and  Services}}",
    ),
    "P": (
        "\hspace{0.2cm}Fertilizer Production",
        "\multicolumn{5}{l}{\\textit{Agricultural Sector: Inputs}}",
    ),
    "W": (
        "\hspace{0.2cm}Water Production",
        "\multicolumn{5}{l}{\\textit{Agricultural Sector: Inputs}}",
    ),
    "Pho": (
        "\hspace{0.2cm}Phosphate Extraction",
        "\multicolumn{5}{l}{\\textit{Extractive Sectors}}",
    ),
    "R": (
        "\hspace{0.2cm}Renewables Production",
        "\multicolumn{5}{l}{\\textit{Extractive Sectors}}",
    ),
    "Fi": (
        "\hspace{0.2cm}Fisheries Production",
        "\multicolumn{5}{l}{\\textit{Others}}",
    ),
    "T": ("\hspace{0.2cm}Timber Production", "\multicolumn{5}{l}{\\textit{Others}}"),
    "Y": ("\hspace{0.2cm}Manufacturing", "\multicolumn{5}{l}{\\textit{Others}}"),
}

def model_variable_result_table(var_desc_dict, price_desc_dict, df_full, df_price_full):
    latex_columns = ['outcome', 'prices', 'outcome_biofuel', 'prices_biofuel'] # , 'outcome_land',  'prices_land'
    
    for key, var_name in var_desc_dict.items():
        if key in table_text.keys():
            price_name = 'p_' + key.split('_')[0]
            price_index_series = df_price_full['index'].str.contains(price_desc_dict[price_name][1], regex=False)
            price_index = price_index_series.index[price_index_series].tolist()[0]
            quant_index_series = df_full['index'].str.contains(var_name[1], regex=False)
            quant_index = quant_index_series.index[quant_index_series].tolist()[0]

            #print(quant_index, key, price_name, price_index, price_desc_dict[price_name][1])
            latex_str = table_text[key][0] # df_full.iloc[quant_index, df_full.columns.get_loc('index')]

            for col_name in latex_columns:
                if 'outcome' in col_name:
                    value = df_full.iloc[quant_index, df_full.columns.get_loc(col_name)]
                else:
                    value = df_price_full.iloc[price_index, df_price_full.columns.get_loc(col_name)]

                latex_str += ' & ' + str(round(value, 3)) 
            latex_str += ' \\\\ '
            #latex_str = re.escape(latex_str)
            typology[table_text[key][1]].append(latex_str)
            
    with open('model_variables_table.txt', 'w') as f:
        for k, values in typology.items():
            print(k + ' \\\\ ')
            f.write(k)
            f.write("\n")
            for v in values:
                print(v)
                f.write(repr(v))
                f.write("\n")
                
    return typology
        
        
    