import pandas as pd

def explain_predictions(explainer, model_input):
    explanations = explainer(
        model_input,
        check_additivity=True,
    )

    shap_values_df = pd.DataFrame(explanations.values, columns=explanations.feature_names)

    all_factors = []
    for row_position in range(len(model_input)):
        factors = []
        for column_position, column in enumerate(shap_values_df.columns):
            factor = {}         
            factor.update({"feature":column,
                        "value":float(model_input.iloc[row_position,column_position]),
                        "shap_value":float(shap_values_df.iloc[row_position,column_position])})  
            factors.append(factor)    

        factors.sort(
        key=lambda factor: abs(factor["shap_value"]),
        reverse=True,
        )
        all_factors.append(factors[:3])
    return all_factors