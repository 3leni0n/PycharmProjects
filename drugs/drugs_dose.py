# Baytril
# From Pep: dilute to 1/10 with sterile saline and administer 0.06

dose = 5/1  # 5mg/kg
trans_gr = 1/1000  # 1kg/1000gr. Transformation to grams, otherwise the weight of the animal should be in kg
animal_weight = 30
concentration = 1/25  # 25mg/ml

iny_vol = dose * trans_gr * animal_weight * concentration
print(iny_vol)


def drug(dose, trans_gr, animal_weight, concentration):
    """
    Get inyected volume of a drug
    :param dose:
    :param trans_gr:
    :param animal_weight:
    :param concentration:
    :return:
    """
    iny_vol = dose * trans_gr * animal_weight * concentration
    return iny_vol