# https://docs.google.com/document/d/1Cy6UkcNtYQc1m331o9gKKZpTWiMkXQxYgeVMecdzUTI/edit#
"""
Phire Tissue Direct PCR Master Mix
Pipetting instructions for a 20 ul reaction (add items in this order). Get an ice bucket to do the pipetting and storing the components in cold temperature during the whole process.

Grin1
H2O: add to 20 ul (7.5 ul)
2X Phire Tissue Direct PCR Master Mix: 10 ul
Primer A: 0.5 ul
Primer B: 0.5 ul
(DNA: 1.5 ul)"""


def pcr_grin1(n_samples, volume=20):

    blank = 1  # For checking contamination
    extra = 1  # For pipetting loss
    n_samples = n_samples + blank + extra  # +2: 1 blanck + 1 for pipetting loss
    print(n_samples)
    print(f'Alicuota for {n_samples - blank - extra} sample(s) + {blank} blank + {extra} extra (for pipetting loss) \n')

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


def pcr_pvcre(n_samples, volume=20):

    blank = 1  # For checking contamination
    extra = 1  # For pipetting loss
    n_samples = n_samples + blank + extra  # +2: 1 blanck + 1 for pipetting loss

    print(n_samples)
    print(f'Alicuota for {n_samples - blank - extra} sample(s) + {blank} blank + {extra} extra (for pipetting loss) \n')

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
