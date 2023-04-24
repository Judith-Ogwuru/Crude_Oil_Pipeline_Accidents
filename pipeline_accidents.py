# Validere take home assignment!
# What qualities of crude oil lead to pipeline accidents?

# Useful python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.style.use("./deeplearning.mplstyle")

# Bring in the Oil Pipeline Accidents Data (pad). Investigate it.
pad = pd.read_excel("data_3.xlsx")
pd.set_option("display.max_columns", None)
#print(pad.head())
#print(pad.shape)
#print(pad.columns)


# The focus is on "crude oil" so filter data to include only crude oil commodity_released_type(cot)
# Also only include columns that are helpful in determining the physical chemical properties of crude oil
# that could be responsible for pipeline accidents.
cot = pad.loc[pad.COMMODITY_RELEASED_TYPE == "CRUDE OIL"]
columns = ["IYEAR", "UNINTENTIONAL_RELEASE_BBLS", "RECOVERED_BBLS", "FATAL", "INJURE", "PRPTY",
           "NUM_PERSONS_HOSP_NOT_OVNGHT", "NUM_INJURED_TREATED_BY_EMT", "CAUSE", "CAUSE_DETAILS",
           "INT_VISUAL_EXAM_RESULTS", "INT_VISUAL_EXAM_DETAILS", "INT_CORROSIVE_COMMODITY_IND",
           "INT_WATER_ACID_IND", "INT_MICROBIOLOGICAL_IND", "INT_EROSION_IND", "INT_OTHER_CORROSION_IND",
           "INT_CORROSION_TYPE_DETAILS", "INCIDENT_UNKNOWN_COMMENTS", "UNKNOWN_SUBTYPE",
           "INTRNL_COR_CORROSIVE_CMDTY_IND", "INTRNL_COR_WTR_DRPOUT_ACID_IND",
           "INTRNL_COR_MICROBIOLOGIC_IND", "INTRNL_COR_EROSION_IND"]
cot = cot[columns]

# Some crude oil details
#print(cot.head())
print(cot.shape)
Total_Injuries = cot["INJURE"].sum() + cot["NUM_PERSONS_HOSP_NOT_OVNGHT"].sum() \
                 + cot["NUM_INJURED_TREATED_BY_EMT"].sum()
Total_Fatalities = cot["FATAL"].sum()
Total_Damage_Cost = cot["PRPTY"].sum()
Total_Barrels_Spilled = cot["UNINTENTIONAL_RELEASE_BBLS"].sum()
Total_Barrels_Recovered = cot["RECOVERED_BBLS"].sum()
print("Crude_Oil_Details")
print(f"Total_Injuries: {Total_Injuries}, "
      f"Total_Fatalities: {Total_Fatalities},"
      f"Total_Damage_Cost: {Total_Damage_Cost}, "
      f"Total_Barrels_Spilled: {Total_Barrels_Spilled},"
      f"Total_Barrels_Recovered: {Total_Barrels_Recovered}")


# From the information provided in the Hazardous Liquid Accident PHMSA F7000 1 Rev 3-2021 Data Fields Pdf document,
# it appears that the physical chemical properties of crude oil that could be responsible for pipeline accidents
# are likely to fall under corrosion failure (cot_cor) and other incident cause (cot_oth) in the cause column.
# Also cause details for corrosion failure is likely to fall under internal corrosion category
# while cause details for other incident type is likely unknown.
# With these in mind, the data can be filtered as follows:
cot_cor = cot.loc[(cot.CAUSE == "CORROSION FAILURE") & (cot.CAUSE_DETAILS == "INTERNAL CORROSION")]
#print(cot_cor.shape)
cot_oth = cot.loc[(cot.CAUSE == "OTHER INCIDENT CAUSE") & (cot.CAUSE_DETAILS == "UNKNOWN")]
#print(cot_oth.shape)



# At this point, I would like time to clean/transform the corrosion failure data
# (removing unwanted columns, updating/filling missing values, renaming columns e.t.c).
#print(cot_cor.head())
#print(cot_cor.shape)
#print(cot_cor.isnull().sum())
cot_cor = cot_cor.copy()
cot_cor[["NUM_PERSONS_HOSP_NOT_OVNGHT", "NUM_INJURED_TREATED_BY_EMT"]] = \
    cot_cor[["NUM_PERSONS_HOSP_NOT_OVNGHT", "NUM_INJURED_TREATED_BY_EMT"]].fillna(0)
cot_cor["FATALITIES"] = cot_cor["FATAL"]
cot_cor["INJURE"] = cot_cor["INJURE"] + cot_cor["NUM_PERSONS_HOSP_NOT_OVNGHT"] + cot_cor["NUM_INJURED_TREATED_BY_EMT"]
#print(cot_cor.loc[cot_cor["INT_VISUAL_EXAM_RESULTS"].isna()])
cot_cor["INT_VISUAL_EXAM_RESULTS"] = cot_cor["INT_VISUAL_EXAM_RESULTS"].fillna("OTHER")
cot_cor.replace({"INT_CORROSIVE_COMMODITY_IND": {"YES": "CORROSIVE COMMODITY"}, "INT_WATER_ACID_IND":
    {"YES": "WATER DROP-OUT/ACID"}, "INT_MICROBIOLOGICAL_IND": {"YES": "MICROBIOLOGICAL"},
                 "INT_EROSION_IND": {"YES": "EROSION"}, "INT_OTHER_CORROSION_IND": {"YES": "OTHER"},
                 "INTRNL_COR_CORROSIVE_CMDTY_IND": {"YES": "CORROSIVE COMMODITY"},
                 "INTRNL_COR_WTR_DRPOUT_ACID_IND": {"YES": "WATER DROP-OUT/ACID"},
                 "INTRNL_COR_MICROBIOLOGIC_IND": {"YES": "MICROBIOLOGICAL"},
                 "INTRNL_COR_EROSION_IND": {"YES": "EROSION"}}, inplace=True)
cot_cor["CAUSE_OF_CORROSION"] = cot_cor.INT_CORROSIVE_COMMODITY_IND.\
    combine_first(cot_cor.INT_WATER_ACID_IND.
                  combine_first(cot_cor.INT_MICROBIOLOGICAL_IND.
                                combine_first(cot_cor.INT_EROSION_IND.
                                              combine_first(cot_cor.INT_OTHER_CORROSION_IND.
                                                            combine_first(cot_cor.INTRNL_COR_CORROSIVE_CMDTY_IND.
                                                                          combine_first(cot_cor.INTRNL_COR_WTR_DRPOUT_ACID_IND.
                                                                                        combine_first(cot_cor.INTRNL_COR_MICROBIOLOGIC_IND.
                                                                                                      combine_first(cot_cor.INTRNL_COR_EROSION_IND))))))))
cot_cor["CAUSE_OF_CORROSION"] = cot_cor["CAUSE_OF_CORROSION"].fillna("OTHER")
cot_cor_year = cot_cor
cot_cor = cot_cor.drop(columns=["IYEAR", "NUM_PERSONS_HOSP_NOT_OVNGHT", "NUM_INJURED_TREATED_BY_EMT", "INT_VISUAL_EXAM_DETAILS",
                                "INT_CORROSION_TYPE_DETAILS", "INCIDENT_UNKNOWN_COMMENTS", "UNKNOWN_SUBTYPE",
                                "FATAL", "INT_CORROSIVE_COMMODITY_IND", "INT_WATER_ACID_IND",
                                "INT_MICROBIOLOGICAL_IND", "INT_EROSION_IND", "INT_OTHER_CORROSION_IND",
                                "INTRNL_COR_CORROSIVE_CMDTY_IND", "INTRNL_COR_WTR_DRPOUT_ACID_IND",
                                "INTRNL_COR_MICROBIOLOGIC_IND", "INTRNL_COR_EROSION_IND"])
cot_cor = cot_cor.rename(columns={"UNINTENTIONAL_RELEASE_BBLS": "BARRELS_SPILLED",
                                  "RECOVERED_BBLS": "BARRELS_RECOVERED", "INJURE": "INJURIES",
                                  "PRPTY": "DAMAGE_COST", "CAUSE": "ACCIDENT_CAUSE", "INT_VISUAL_EXAM_RESULTS":
                                      "VISUAL_EXAMINATION_RESULTS"})

print(cot_cor.shape)
# Some corrosion failure/internal corrosion details
Total_Injuries = cot_cor["INJURIES"].sum()
Total_Fatalities = cot_cor["FATALITIES"].sum()
Total_Damage_Cost = cot_cor["DAMAGE_COST"].sum()
Total_Barrels_Spilled = cot_cor["BARRELS_SPILLED"].sum()
Total_Barrels_Recovered = cot_cor["BARRELS_RECOVERED"].sum()
print("Corrosion_Failure_Internal_Corrosion_Details")
print(f"Total_Injuries: {Total_Injuries}, "
      f"Total_Fatalities: {Total_Fatalities},"
      f"Total_Damage_Cost: {Total_Damage_Cost}, "
      f"Total_Barrels_Spilled: {Total_Barrels_Spilled},"
      f"Total_Barrels_Recovered: {Total_Barrels_Recovered}")
#print(cot_cor)



# Grouping by Visual Examination Results and plotting results.
cor_ver = cot_cor.groupby("VISUAL_EXAMINATION_RESULTS").sum().reset_index()
print(cor_ver)
# Summary statistics of quantitative features.
print(cor_ver.describe())
# Set the index to be the 'Visual Examination Results' column
cor_ver.set_index("VISUAL_EXAMINATION_RESULTS", inplace=True)
# Create a figure with two subplots.
# The first is a chart of Amount(BBLS) vs Visual Examination Results.
# The second is a chart of Damage Cost vs Visual Examination Results.
fig_1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# Plot Barrels Spilled and Barrels Recovered on the first subplot
cor_ver[["BARRELS_SPILLED", "BARRELS_RECOVERED"]].plot(kind='bar', ax=ax1, color=['cyan', 'green'])
ax1.set_title('Barrels Spilled and Recovered by Visual Examination Results')
ax1.set_xlabel("Visual Examination Results")
ax1.set_ylabel("Amount (BBLS)")
# Plot Damage Cost on the second subplot
cor_ver['DAMAGE_COST'].plot(kind='bar', ax=ax2, color='purple')
ax2.set_title('Damage Cost by Visual Examination Results')
ax2.set_xlabel("Visual Examination Results")
ax2.set_ylabel("Damage Cost (USD)")
# Add legend to the first subplot
ax1.legend(["Barrels Spilled", "Barrels Recovered"])
# Adjust layout
plt.tight_layout()



# Grouping by Cause of Corrosion and plotting results.
cor_coc = cot_cor.groupby("CAUSE_OF_CORROSION").sum().reset_index()
print(cor_coc)
print(cor_coc.describe())
# Set the index to be the 'CAUSE_OF_CORROSION' column
cor_coc.set_index("CAUSE_OF_CORROSION", inplace=True)
# Create a figure with two subplots
fig_2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# Plot Barrels Spilled and Barrels Recovered on the first subplot
cor_coc[["BARRELS_SPILLED", "BARRELS_RECOVERED"]].plot(kind='bar', ax=ax1, color=['cyan', 'green'])
ax1.set_title('Barrels Spilled and Recovered by Cause of Corrosion')
ax1.set_xlabel("Cause of Corrosion")
ax1.set_ylabel("Amount (BBLS)")
# Plot Damage Cost on the second subplot
cor_coc['DAMAGE_COST'].plot(kind='bar', ax=ax2, color='purple')
ax2.set_title('Damage Cost by Cause of Corrosion')
ax2.set_xlabel("Cause of Corrosion")
ax2.set_ylabel("Damage Cost (USD)")
# Add legend to the first subplot
ax1.legend(["Barrels Spilled", "Barrels Recovered"])
# Adjust layout
plt.tight_layout()



# Grouping by Visual Examination Result and Cause of Corrosion. The plots are shown below.
cor_vecc = cot_cor.groupby(["VISUAL_EXAMINATION_RESULTS", "CAUSE_OF_CORROSION"]).sum().reset_index()
print(cor_vecc)
print(cor_vecc.describe())
# Set the index to be the combination of 'VISUAL_EXAMINATION_RESULTS' and 'CAUSE_OF_CORROSION' columns
cor_vecc.set_index(['VISUAL_EXAMINATION_RESULTS', 'CAUSE_OF_CORROSION'], inplace=True)
# Create a figure with subplots
fig_3, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
# Plot 'Barrels Silled and Barrels Recovered on the first subplot
cor_vecc["BARRELS_SPILLED"].plot(kind='bar', ax=ax1, color='cyan', label='Barrels Spilled', stacked=True)
cor_vecc['BARRELS_RECOVERED'].plot(kind='bar', ax=ax1, color='green', label='Barrels Recovered', stacked=True)
cor_vecc['DAMAGE_COST'].plot(kind='bar', ax=ax2, color='purple', stacked=True)
# Set the chart title, x-axis label, and y-axis label for both subplots
ax1.set_title('Barrels Spilled and Recovered by Visual Examination Results / Cause of Corrosion')
ax1.set_xlabel('Visual Examination Results / Cause of Corrosion')
ax1.set_ylabel("Amount (BBLS)")
ax2.set_title('Damage Cost by Visual Examination Results / Cause of Corrosion')
ax2.set_xlabel('Visual Examination Results / Cause of Corrosion')
ax2.set_ylabel("Damage Cost (USD)")
# Add legend to the first subplot
ax1.legend()
# Adjust layout and display the plot
plt.tight_layout()





# Now is the time to clean/transform the other incident cause data
# (dropping unwanted columns, updating/filling missing values, renaming columns e.t.c).
#print(cot_oth.head())
#print(cot_oth.shape)
#print(cot_oth.isnull().sum())
cot_oth = cot_oth.copy()
cot_oth[["NUM_PERSONS_HOSP_NOT_OVNGHT", "NUM_INJURED_TREATED_BY_EMT"]] = \
    cot_oth[["NUM_PERSONS_HOSP_NOT_OVNGHT", "NUM_INJURED_TREATED_BY_EMT"]].fillna(0)
cot_oth["FATALITIES"] = cot_oth["FATAL"]
cot_oth["INJURE"] = cot_oth["INJURE"] + cot_oth["NUM_PERSONS_HOSP_NOT_OVNGHT"] + cot_oth["NUM_INJURED_TREATED_BY_EMT"]
cot_oth = cot_oth.drop(columns=["IYEAR", "INT_VISUAL_EXAM_RESULTS", "INT_VISUAL_EXAM_DETAILS",
                                "INT_CORROSIVE_COMMODITY_IND", "INT_WATER_ACID_IND", "INT_MICROBIOLOGICAL_IND",
                                "INT_EROSION_IND", "INT_OTHER_CORROSION_IND", "INT_CORROSION_TYPE_DETAILS",
                                "INCIDENT_UNKNOWN_COMMENTS", "NUM_PERSONS_HOSP_NOT_OVNGHT",
                                "NUM_INJURED_TREATED_BY_EMT", "FATAL", "INTRNL_COR_CORROSIVE_CMDTY_IND",
                                "INTRNL_COR_WTR_DRPOUT_ACID_IND", "INTRNL_COR_MICROBIOLOGIC_IND",
                                "INTRNL_COR_EROSION_IND"])
cot_oth = cot_oth.rename(columns={"UNINTENTIONAL_RELEASE_BBLS": "BARRELS_SPILLED",
                                  "RECOVERED_BBLS": "BARRELS_RECOVERED", "INJURE": "INJURIES",
                                  "PRPTY": "DAMAGE_COST", "CAUSE": "ACCIDENT_CAUSE", "UNKNOWN_SUBTYPE":
                                  "UNKNOWN_DETAILS"})
print(cot_oth.shape)
# Some other incident cause/unknown details
Total_Injuries = cot_oth["INJURIES"].sum()
Total_Fatalities = cot_oth["FATALITIES"].sum()
Total_Damage_Cost = cot_oth["DAMAGE_COST"].sum()
Total_Barrels_Spilled = cot_oth["BARRELS_SPILLED"].sum()
Total_Barrels_Recovered = cot_oth["BARRELS_RECOVERED"].sum()
print("Other_Incident_Cause_Unknown_Details")
print(f"Total_Injuries: {Total_Injuries}, "
      f"Total_Fatalities: {Total_Fatalities},"
      f"Total_Damage_Cost: {Total_Damage_Cost}, "
      f"Total_Barrels_Spilled: {Total_Barrels_Spilled},"
      f"Total_Barrels_Recovered: {Total_Barrels_Recovered}")

# Grouping by Unknown Details, Mapping and plotting results.
oth_unk = cot_oth.groupby("UNKNOWN_DETAILS").sum().reset_index()
# Define a mapping between old values and new values
mapping = {"\"INVESTIGATION COMPLETE, CAUSE OF ACCIDENT UNKNOWN\"": "IC/COAU",
           "\"STILL UNDER INVESTIGATION, CAUSE OF ACCIDENT TO BE DETERMINED" + "* (*SUPPLEMENTAL REPORT REQUIRED)\"":
               "SUI/COAD"}
oth_unk = oth_unk.replace(mapping)
print(oth_unk)
print(oth_unk.describe())
# Set the index to be the 'Unknown Details' column
oth_unk.set_index("UNKNOWN_DETAILS", inplace=True)
# Create a figure with two subplots.
# The first is a chart of Barrels Spilled and Recovered by Unknown Details.
# The second is a chart of Damage Cost by Unknown Details.
# The third is a chart of 'Injuries / Fatalities by Unknown Details'.
fig_4, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
# Plot Barrels Spilled and Barrels Recovered on the first subplot
oth_unk[["BARRELS_SPILLED", "BARRELS_RECOVERED"]].plot(kind='bar', ax=ax1, color=['cyan', 'green'])
ax1.set_title('Barrels Spilled and Recovered by Unknown Details')
ax1.set_xlabel("Unknown Details")
ax1.set_ylabel("Amount (BBLS)")
# Plot Damage Cost on the second subplot
oth_unk['DAMAGE_COST'].plot(kind='bar', ax=ax2, color='purple')
ax2.set_title('Damage Cost by Unknown Details')
ax2.set_xlabel("Unknown Details")
ax2.set_ylabel("Damage Cost (USD)")
# Plot Injuries and Fatalities on the third subplot
oth_unk[["INJURIES", "FATALITIES"]].plot(kind='bar', ax=ax3)
ax3.set_title('Injuries / Fatalities by Unknown Details')
ax3.set_xlabel("Unknown Details")
ax3.set_ylabel("No of Persons")
# Add legend to the first and third subplots
ax1.legend(["Barrels Spilled", "Barrels Recovered"])
ax3.legend(["Injuries", "Fatalities"])
# Adjust layout
plt.tight_layout()
plt.show()


# Summary of Results for internal corrosion/corrosion failure/crude oil
accident_cor = (len(cot_cor) / len(pad)) * 100
barrels_spilled_cor = (cot_cor["BARRELS_SPILLED"].sum() / pad["UNINTENTIONAL_RELEASE_BBLS"].sum()) * 100
barrels_recovered_cor = (cot_cor["BARRELS_RECOVERED"].sum() / pad["RECOVERED_BBLS"].sum()) * 100
injuries_cor = (cot_cor["INJURIES"].sum() / (pad["INJURE"].sum() + pad["NUM_PERSONS_HOSP_NOT_OVNGHT"].sum()
                                             + pad["NUM_INJURED_TREATED_BY_EMT"].sum())) * 100
fatalities_cor = (cot_cor["FATALITIES"].sum() / pad["FATAL"].sum()) * 100
damage_cor = (cot_cor["DAMAGE_COST"].sum() / pad["PRPTY"].sum()) * 100
print(f"% of accidents caused by internal corrosion: {accident_cor}")
print(f"% of unintentional barrels spilled from internal corrosion: {barrels_spilled_cor}")
print(f"% of barrels recovered from internal corrosion: {barrels_recovered_cor}")
print(f"% of injuries caused by internal corrosion: {injuries_cor}")
print(f"% of fatalities caused by internal corrosion: {fatalities_cor}")
print(f"% of property damage cost from internal corrosion: {damage_cor}")

# Summary of Results for unknown/other incident cause/crude oil
accident_oth = (len(cot_oth) / len(pad)) * 100
barrels_spilled_oth = (cot_oth["BARRELS_SPILLED"].sum() / pad["UNINTENTIONAL_RELEASE_BBLS"].sum()) * 100
barrels_recovered_oth = (cot_oth["BARRELS_RECOVERED"].sum() / pad["RECOVERED_BBLS"].sum()) * 100
injuries_oth = (cot_oth["INJURIES"].sum() / (pad["INJURE"].sum() + pad["NUM_PERSONS_HOSP_NOT_OVNGHT"].sum()
                                             + pad["NUM_INJURED_TREATED_BY_EMT"].sum())) * 100
fatalities_oth = (cot_oth["FATALITIES"].sum() / pad["FATAL"].sum()) * 100
damage_oth = (cot_oth["DAMAGE_COST"].sum() / pad["PRPTY"].sum()) * 100
print(f"% of accidents caused by unknown: {accident_oth}")
print(f"% of unintentional barrels spilled from unknown: {barrels_spilled_oth}")
print(f"% of barrels recovered from unknown: {barrels_recovered_oth}")
print(f"% of injuries caused by unknown: {injuries_oth}")
print(f"% of fatalities caused by unknown: {fatalities_oth}")
print(f"% of property damage cost from unknown: {damage_oth}")


# Part B, Question 2
# Have we gotten better at regulating those properties over time?
cot_cor_year = cot_cor_year.copy().rename(columns={"IYEAR": "YEAR", "PRPTY": "DAMAGE_COST"})
cost_year = cot_cor_year.groupby("YEAR").DAMAGE_COST.sum().reset_index()
first_7year_ave = cost_year.iloc[:7, 1].sum()
last_7year_ave = cost_year.iloc[7:, 1].sum()
print(f"Average cost for the first seven years : {first_7year_ave}")
print(f"Average cost for the last seven years : {last_7year_ave}")

