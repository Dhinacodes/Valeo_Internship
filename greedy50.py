import streamlit as st
import pandas as pd
import numpy as np
import io

def normalize_rowwise(value, corresponding_value, maximize=True):
    """
    Normalize a value based on its corresponding value in the same row.
    If maximize is True, higher values are better.
    If maximize is False, lower values are better.
    """
    if maximize:
        return (value - corresponding_value) / abs(value - corresponding_value)
    else:
        return (corresponding_value - value) / abs(value - corresponding_value)

def greedy_search(df):
    """
    Greedy search to optimize PWM technique based on row-wise normalization.
    """
    criteria_3P = ["3P Power loss (kw)", "3P Torque pulse (%)", "3P Power Delivery"]
    criteria_2P = ["2P Power loss (kw)", "2P Torque pulse (%)", "2P Power delivery"]

    # Results storage
    optimized_results = []

    # Equal weights for all criteria
    weight = 1

    # Iterate through each row
    for _, row in df.iterrows():
        scores_3P = []
        scores_2P = []

        # Normalize and calculate weighted scores
        for c3, c2, maximize in zip(criteria_3P, criteria_2P, [False, False, True]):
            norm_3P = normalize_rowwise(row[c3], row[c2], maximize=maximize)
            norm_2P = normalize_rowwise(row[c2], row[c3], maximize=maximize)
            scores_3P.append(weight * norm_3P)
            scores_2P.append(weight * norm_2P)

        # Sum the scores for 3P and 2P
        total_score_3P = sum(scores_3P)
        total_score_2P = sum(scores_2P)

        # Choose the better PWM technique
        optimized_results.append("3P" if total_score_3P > total_score_2P else "2P")

    # Add results to the dataframe
    df["PWM Technique"] = optimized_results
    return df[["speed", "PWM Technique"]]

def main():
    st.title("Greedy Search Algorithm for PWM Optimization")
    st.write("Upload your dataset in Excel format")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)

            # Process the dataset using greedy search
            result = greedy_search(df)

            # Display the result
            st.subheader("Optimized PWM Techniques (Greedy Search Algorithm)")
            st.write(result)

            # Save the result to an Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result.to_excel(writer, index=False, sheet_name='Optimized PWM Techniques')
            output.seek(0)

            # Download the result as Excel
            st.download_button(
                label="Download Result as Excel",
                data=output,
                file_name="optimized_pwm_greedy_search.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
