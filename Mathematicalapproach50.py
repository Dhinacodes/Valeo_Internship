import streamlit as st
import pandas as pd
import numpy as np
import io
import os


def normalize_rowwise(value, reference, maximize=True):
    """
    Normalize a value based on a reference value from an adjacent column.
    If `maximize` is True, higher values are better.
    If `maximize` is False, lower values are better.
    """
    if maximize:
        return (value - min(value, reference)) / (max(value, reference) - min(value, reference))
    else:
        return (max(value, reference) - value) / (max(value, reference) - min(value, reference))


def calculate_tchebycheff(row, ideal_values, weights):
    """
    Calculate the Tchebycheff norm for a given row.
    """
    deviations = [weights[i] * abs(row[i] - ideal_values[i]) for i in range(len(row))]
    return max(deviations)


def calculate_compromise_distance(row, ideal_values, ranges, weights, p=1):
    """
    Calculate the Compromise (Manhattan or Euclidean) distance for a given row.
    """
    deviations = [(row[i] - ideal_values[i]) / ranges[i] for i in range(len(row))]
    if p == 1:  # Manhattan distance
        return sum(weights[i] * abs(deviations[i]) for i in range(len(deviations)))
    elif p == 2:  # Euclidean distance
        return np.sqrt(sum(weights[i] * deviations[i] ** 2 for i in range(len(deviations))))


def process_dataset(df, methodology):
    # Criteria columns for 3P and 2P
    criteria_3P = ["3P Power loss (kw)", "3P Torque pulse (%)", "3P Power Delivery"]
    criteria_2P = ["2P Power loss (kw)", "2P Torque pulse (%)", "2P Power delivery"]

    # Normalize the data row-wise
    for col_3p, col_2p in zip(criteria_3P, criteria_2P):
        if "Power loss" in col_3p or "Torque pulse" in col_3p:
            df[f"norm_{col_3p}"] = df.apply(lambda row: normalize_rowwise(row[col_3p], row[col_2p], maximize=False), axis=1)
            df[f"norm_{col_2p}"] = df.apply(lambda row: normalize_rowwise(row[col_2p], row[col_3p], maximize=False), axis=1)
        else:
            df[f"norm_{col_3p}"] = df.apply(lambda row: normalize_rowwise(row[col_3p], row[col_2p], maximize=True), axis=1)
            df[f"norm_{col_2p}"] = df.apply(lambda row: normalize_rowwise(row[col_2p], row[col_3p], maximize=True), axis=1)

    # Ideal values (0 for losses and pulses, 1 for delivery)
    ideal_3P = [0, 0, 1]
    ideal_2P = [0, 0, 1]

    # Equal weights for all criteria
    weights = [1, 1, 1]
    ranges_3P = [1] * len(criteria_3P)  # Since normalization is row-wise, the range is already scaled between 0 and 1
    ranges_2P = [1] * len(criteria_2P)

    # Process based on selected methodology
    if methodology == "Tchebycheff":
        # Calculate Tchebycheff Norms for 3P and 2P
        df["Tchebycheff_3P"] = df.apply(lambda row: calculate_tchebycheff(
            [row[f"norm_{col}"] for col in criteria_3P], ideal_3P, weights), axis=1)

        df["Tchebycheff_2P"] = df.apply(lambda row: calculate_tchebycheff(
            [row[f"norm_{col}"] for col in criteria_2P], ideal_2P, weights), axis=1)

        # Determine which configuration is optimized
        df["PWM Technique"] = df.apply(lambda row: "3P" if row["Tchebycheff_3P"] < row["Tchebycheff_2P"] else "2P", axis=1)

    elif methodology == "Compromise":
        # Calculate Compromise distances for 3P and 2P
        df["Compromise_3P"] = df.apply(lambda row: calculate_compromise_distance(
            [row[f"norm_{col}"] for col in criteria_3P], ideal_3P, ranges_3P, weights, p=1), axis=1)

        df["Compromise_2P"] = df.apply(lambda row: calculate_compromise_distance(
            [row[f"norm_{col}"] for col in criteria_2P], ideal_2P, ranges_2P, weights, p=1), axis=1)

        # Determine which configuration is optimized
        df["PWM Technique"] = df.apply(lambda row: "3P" if row["Compromise_3P"] < row["Compromise_2P"] else "2P", axis=1)

    elif methodology == "Combined":
        # Calculate Tchebycheff Norms for 3P and 2P
        df["Tchebycheff_3P"] = df.apply(lambda row: calculate_tchebycheff(
            [row[f"norm_{col}"] for col in criteria_3P], ideal_3P, weights), axis=1)

        df["Tchebycheff_2P"] = df.apply(lambda row: calculate_tchebycheff(
            [row[f"norm_{col}"] for col in criteria_2P], ideal_2P, weights), axis=1)

        # Calculate Compromise distances for 3P and 2P
        df["Compromise_3P"] = df.apply(lambda row: calculate_compromise_distance(
            [row[f"norm_{col}"] for col in criteria_3P], ideal_3P, ranges_3P, weights, p=1), axis=1)

        df["Compromise_2P"] = df.apply(lambda row: calculate_compromise_distance(
            [row[f"norm_{col}"] for col in criteria_2P], ideal_2P, ranges_2P, weights, p=1), axis=1)

        # Calculate combined score using both methods
        df["Combined_3P"] = (df["Tchebycheff_3P"] + df["Compromise_3P"]) / 2
        df["Combined_2P"] = (df["Tchebycheff_2P"] + df["Compromise_2P"]) / 2

        # Determine which configuration is optimized
        df["PWM Technique"] = df.apply(lambda row: "3P" if row["Combined_3P"] < row["Combined_2P"] else "2P", axis=1)

    # Return only speed and PWM Technique
    return df[["speed", "PWM Technique"]]


def main():
    st.title("PWM Optimization Using Different Methodologies")

    st.write("Upload your dataset in Excel format")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    methodology = st.selectbox("Select Optimization Methodology", ["Tchebycheff", "Compromise", "Combined"])

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)

            # Process the dataset based on the selected methodology
            result = process_dataset(df, methodology)

            # Display the result
            st.subheader(f"Optimized PWM Techniques - {methodology} Methodology")
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
                file_name="optimized_pwm_techniques.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
