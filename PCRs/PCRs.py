# https://docs.google.com/document/d/1Cy6UkcNtYQc1m331o9gKKZpTWiMkXQxYgeVMecdzUTI/edit#

def pcr_grin1_thermo(n_samples, volume=20, control=True, extra=1):
    """
    Phire Tissue Direct PCR Master Mix (https://www.thermofisher.com/order/catalog/product/F170S)
    :param n_samples:
    :param volume:
    :param control:
    :param extra: For pipetting loss
    :return:
    """

    if control:
        control_negative = 1
        control_positive = 1
        control_hemi = 1
        blank = 1  # For checking contamination
        n_samples = n_samples + blank + extra + control_negative + control_positive + control_hemi
    else:
        n_samples = n_samples + extra

    print(f'Alicuota for {n_samples - blank - extra - control_negative - control_positive - control_hemi} sample(s) + '
          f'{blank} blank + {extra} extra + {control_negative + control_positive + control_hemi} controls \n')

    print('Add to the mix:')
    water = 7.5 * n_samples  # Nuclease free, add till complete 20 ul (7.5 ul)
    print(f'Add {water} ul of nuclease free water')
    phire_tissue_direct_pcr_master_mix_2x = 10 * n_samples
    print(f'Add {phire_tissue_direct_pcr_master_mix_2x} ul of 2X Phire Tissue Direct PCR Master Mix')
    primer_in = 0.5 * n_samples  # 100 uM concentration
    print(f'Add {primer_in} ul of primer in (100 uM)')
    primer_out = 0.5 * n_samples  # 100 uM concentration
    print(f'Add {primer_out} ul of primer out (100 uM)\n')
    dna = 1.5

    print('Add to each PCR tube:')
    print(f'Add {7.5 + 10 + 0.5 + 0.5} ul of mix')
    print(f'Add {dna} ul of DNA')

    assert(((water + phire_tissue_direct_pcr_master_mix_2x + primer_in + primer_out) / n_samples) + dna) == volume


def pcr_grin1_takara(n_samples, volume=25, control=True, extra=1):

    if control:
        control_negative = 1
        control_positive = 1
        control_hemi = 1
        blank = 1  # For checking contamination
        n_samples = n_samples + blank + extra + control_negative + control_positive + control_hemi
    else:
        n_samples = n_samples + extra  # For pipetting loss

    print(f'Alicuota for {n_samples - blank - extra - control_negative - control_positive - control_hemi} sample(s) + '
          f'{blank} blank + {extra} extra + {control_negative + control_positive + control_hemi} controls \n')

    print('Add to the mix:')
    water = 16.3 * n_samples  # Nuclease free, add till complete 20 ul (7.5 ul)
    print(f'Add {water} ul of nuclease free water')
    dNTP = 3 * n_samples
    print(f'Add {dNTP} ul of dNTP')
    buffer_10x = 2.5 * n_samples
    print(f'Add {buffer_10x} ul of buffer 10x')
    primer_in = 0.5 * n_samples  # 100 uM concentration
    print(f'Add {primer_in} ul of primer in (100 uM)')
    primer_out = 0.5 * n_samples  # 100 uM concentration
    print(f'Add {primer_out} ul of primer out (100 uM)')
    taq = 0.2 * n_samples  # 50 uM concentration
    print(f'Add {taq} ul of Taq Polimerase\n')
    dna = 2

    print('Add to each PCR tube:')
    print(f'Add {16.3 + 3 + 2.5 + 0.5 + 0.5 + 0.2} ul of mix')
    print(f'Add {dna} ul of DNA')

    assert(((water + dNTP + buffer_10x + primer_in + primer_out + taq) / n_samples) + dna) == volume


def pcr_pvcre_thermo(n_samples, volume=20, control=True, extra=1):
    """
    Phire Tissue Direct PCR Master Mix (https://www.thermofisher.com/order/catalog/product/F170S)
    :param n_samples:
    :param volume:
    :param control:
    :param extra: # For pipetting loss
    :return:
    """

    if control:
        control_negative = 1
        control_positive = 1
        control_hemi = 1
        blank = 1  # For checking contamination
        n_samples = n_samples + blank + extra + control_negative + control_positive + control_hemi
    else:
        n_samples = n_samples + extra

    print(f'Alicuota for {n_samples - blank - extra - control_negative - control_positive - control_hemi} sample(s) + '
          f'{blank} blank + {extra} extra + {control_negative + control_positive + control_hemi} controls \n')

    print('Add to the mix:')
    water = 5 * n_samples  # Nuclease free, add till complete 20 ul (7.5 ul)
    print(f'Add {water} ul of nuclease free water')
    phire_tissue_direct_pcr_master_mix_2x = 10 * n_samples
    print(f'Add {phire_tissue_direct_pcr_master_mix_2x} ul of 2X Phire Tissue Direct PCR Master Mix')
    primer_A = 1 * n_samples  # 50 uM concentration
    print(f'Add {primer_A} ul of primer A (50 uM)')
    primer_B = 1 * n_samples  # 50 uM concentration
    print(f'Add {primer_B} ul of primer B (50 uM)')
    primer_C = 1 * n_samples  # 50 uM concentration
    print(f'Add {primer_C} ul of primer C (50 uM)')
    primer_D = 1 * n_samples  # 50 uM concentration
    print(f'Add {primer_D} ul of primer D (50 uM)\n')
    dna = 1

    print('Add to each PCR tube:')
    print(f'Add {5 + 10 + 1 + 1 + 1 + 1} ul of mix')
    print(f'Add {dna} ul of DNA')

    assert(((water + phire_tissue_direct_pcr_master_mix_2x + primer_A + primer_B + primer_C + primer_D) / n_samples) +
           dna) == volume


def pcr_pvcre_takara(n_samples, volume=25, control=True, extra=1):

    if control:
        control_negative = 1
        control_positive = 1
        control_hemi = 1
        blank = 1  # For checking contamination
        n_samples = n_samples + blank + extra + control_negative + control_positive + control_hemi
    else:
        n_samples = n_samples + extra  # For pipetting loss

    print(f'Alicuota for {n_samples - blank - extra - control_negative - control_positive - control_hemi} sample(s) + '
          f'{blank} blank + {extra} extra + {control_negative + control_positive + control_hemi} controls \n')

    print('Add to the mix:')
    water = 13.8 * n_samples  # Nuclease free, add till complete 20 ul (7.5 ul)
    print(f'Add {water} ul of nuclease free water')
    dNTP = 3 * n_samples
    print(f'Add {dNTP} ul of dNTP')
    buffer_10x = 2.5 * n_samples
    print(f'Add {buffer_10x} ul of buffer 10x')
    primer_A = 1 * n_samples  # 50 uM concentration
    print(f'Add {primer_A} ul of primer A (50 uM)')
    primer_B = 1 * n_samples  # 50 uM concentration
    print(f'Add {primer_B} ul of primer B (50 uM)')
    primer_C = 1 * n_samples  # 50 uM concentration
    print(f'Add {primer_C} ul of primer C (50 uM)')
    primer_D = 1 * n_samples  # 50 uM concentration
    print(f'Add {primer_D} ul of primer D (50 uM)')
    taq = 0.2 * n_samples  # 50 uM concentration
    print(f'Add {taq} ul of Taq Polimerase\n')
    dna = 1.5

    print('Add to each PCR tube:')
    print(f'Add {13.8 + 3 + 2.5 + 1 + 1 + 1 + 1 + 0.2} ul of mix')
    print(f'Add {dna} ul of DNA')

    assert(((water + dNTP + buffer_10x + primer_A + primer_B + primer_C + primer_D + taq) / n_samples) + dna) == volume
