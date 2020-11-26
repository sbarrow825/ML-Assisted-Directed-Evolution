import shelve
import pandas as pd

print("reading excels")

screenedVariants_df = pd.read_excel("GB1screenedvariants.xlsx")
fittedVariants_df = pd.read_excel("GB1fittedvariants.xlsx")

print("converting excel columns to lists")

screenedVariants_list = list(screenedVariants_df.to_dict()["Variants"].values())
fittedVariants_list = list(fittedVariants_df.to_dict()["Variants"].values())

screenedVariants_fitesses = list(screenedVariants_df.to_dict()["Fitness"].values())
fittedVariants_fitnesses = list(fittedVariants_df.to_dict()["Imputed fitness"].values())

GB1dataset_fitted = shelve.open("GB1dataset_fitted")

print("collecting screened variant data")
screenedCount = 0
screenedTotal = len(screenedVariants_list)

for screenedVariant, fitness in zip(screenedVariants_list, screenedVariants_fitesses):
    GB1dataset_fitted[screenedVariant] = fitness
    screenedCount += 1
    if not screenedCount % 1000:
        print('finished {0} of the {1} screened variants'.format(screenedCount, screenedTotal))

print("collecting fitted variant data")
fittedCount = 0
fittedTotal = len(fittedVariants_list)

for fittedVariant, fitness in zip(fittedVariants_list, fittedVariants_fitnesses):
    GB1dataset_fitted[fittedVariant] = fitness
    fittedCount += 1
    if not fittedCount % 1000:
        print('finished {0} of the {1} fitted variants'.format(fittedCount, fittedTotal))

GB1dataset_fitted.close()

print("done")
