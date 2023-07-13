import numpy as np
import pandas as pd

#### ####

def get_rates_from_split(split):
    ordered_flag = split['outcomes']
    out = np.zeros((len(ordered_flag), 4))
    out[np.where(ordered_flag == 'hit')[0]] = [1,0,0,0]
    out[np.where(ordered_flag == 'miss')[0]] = [0,1,0,0]
    out[np.where(ordered_flag == 'cr')[0]] = [0,0,1,0]
    out[np.where(ordered_flag == 'fa')[0]] = [0,0,0,1]
    out = out.astype(int)
    return out, split['trial_index'].to_numpy()[-1]


def get_hr_fa_acc(r_):
    hit_rate = r_[0] / (np.sum((r_[0], r_[1]), 0) + 0.0001)
    fa_rate = r_[3] / (np.sum((r_[2], r_[3]), 0) + 0.0001)
    accuracy = (r_[0] + r_[2]) / r_.sum()
    
    return hit_rate, fa_rate, accuracy



################ used #####################

def chunk_data(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def split_splits_by_task_type(split, target_task_type = 'combined task'):

    by_animal = []
    for animal in split:
        by_day = []
        for day in animal:
            if day['task_types'].iloc[0] == target_task_type:
                by_day.append(day)
            else:
                continue
        by_animal.append(by_day)
        
    return by_animal


def get_vals2(split, individual_animal_index = None):
    
    ## make over_animals
    over_animals = []
    
    if not individual_animal_index:
        for a in split[0:]:
            over_days = []
            for day in a:
                if not day.shape[0]:
                    continue
                out, tx = get_rates_from_split(day)
                out = out.sum(0)
                hr, fa, acc = get_hr_fa_acc(out)
                over_days.append([hr, fa, acc])

            over_animals.append(over_days)
    else:
        for a in split[individual_animal_index: individual_animal_index+1]:
            over_days = []
            for day in a:
                if not day.shape[0]:
                    continue
                out, tx = get_rates_from_split(day)
                out = out.sum(0)
                hr, fa, acc = get_hr_fa_acc(out)
                over_days.append([hr, fa, acc])

            over_animals.append(over_days)
        
        
    
    ## make stat_array
    max_len = np.max([len(i) for i in over_animals])
    stat_array = np.empty((len(over_animals), max_len, 3)) * np.nan

    for anim_ind in range(len(stat_array)):
        append_array = np.array(over_animals[anim_ind])
        stat_array[anim_ind, 0:len(append_array), :] = np.array(append_array)
        
    # get number of subjects active at each split value
    active_animal = np.zeros((len(stat_array), stat_array.shape[1]))
    for i in range(len(stat_array)):
        for j in range(active_animal.shape[1]):
            if not np.isnan(stat_array[i, j][0]):
                active_animal[i, j] = 1
        
    hr, fa, acc = np.nanmean(stat_array, 0).T
    hr_sem, fa_sem, acc_sem = (np.nanstd(stat_array, 0) / np.sqrt(np.tile(active_animal.sum(0).reshape(-1, 1), 3))).T
    return hr, fa, acc, hr_sem, fa_sem, acc_sem



def get_vals_foil_type(split, individual_animal_index = None):
    over_animals = []
    
    if type(individual_animal_index) != int:
        for a in split:

            over_days_ru = []
            over_days_rd = []
            over_days_lu = []
            over_days_ld = []
            #print(len(a))
            for day in a:
                day

                serial_rate = day.groupby('serial').apply(get_rates_from_split)
                out_ru, tx_ru = serial_rate.ru
                out_rd, tx_rd = serial_rate.rd
                out_lu, tx_lu = serial_rate.lu
                out_ld, tx_ld = serial_rate.ld

                hr_ru, fa_ru, acc_ru = get_hr_fa_acc(out_ru.sum(0))
                hr_rd, fa_rd, acc_rd = get_hr_fa_acc(out_rd.sum(0))
                hr_lu, fa_lu, acc_lu = get_hr_fa_acc(out_lu.sum(0))
                hr_ld, fa_ld, acc_ld = get_hr_fa_acc(out_ld.sum(0))

                over_days_ru.append([hr_ru, fa_ru, acc_ru])
                over_days_rd.append([hr_rd, fa_rd, acc_rd])
                over_days_lu.append([hr_lu, fa_lu, acc_lu])
                over_days_ld.append([hr_ld, fa_ld, acc_ld])

            over_animals.append([over_days_ru, over_days_rd, over_days_lu, over_days_ld])
            
    else:
        for a in split[individual_animal_index:individual_animal_index+1]:

            over_days_ru = []
            over_days_rd = []
            over_days_lu = []
            over_days_ld = []
            for day in a:
                

                serial_rate = day.groupby('serial').apply(get_rates_from_split)
                out_ru, tx_ru = serial_rate.ru
                out_rd, tx_rd = serial_rate.rd
                out_lu, tx_lu = serial_rate.lu
                out_ld, tx_ld = serial_rate.ld

                hr_ru, fa_ru, acc_ru = get_hr_fa_acc(out_ru.sum(0))
                hr_rd, fa_rd, acc_rd = get_hr_fa_acc(out_rd.sum(0))
                hr_lu, fa_lu, acc_lu = get_hr_fa_acc(out_lu.sum(0))
                hr_ld, fa_ld, acc_ld = get_hr_fa_acc(out_ld.sum(0))

                over_days_ru.append([hr_ru, fa_ru, acc_ru])
                over_days_rd.append([hr_rd, fa_rd, acc_rd])
                over_days_lu.append([hr_lu, fa_lu, acc_lu])
                over_days_ld.append([hr_ld, fa_ld, acc_ld])

            over_animals.append([over_days_ru, over_days_rd, over_days_lu, over_days_ld])



    max_len = np.max([len(i[0]) for i in over_animals])
    stat_array = np.empty((len(over_animals), 4, max_len, 3)) * np.nan

    for anim_ind in range(len(stat_array)):
        append_array = np.array(over_animals[anim_ind])
        stat_array[anim_ind, :, 0:append_array.shape[1], :] = np.array(append_array)


    # get number of subjects active at each split value
    active_animal = np.zeros((len(stat_array), stat_array.shape[2]))
    for i in range(len(stat_array)):
        for j in range(active_animal.shape[1]):
            if not np.isnan(stat_array[i, :, j, :][0][0]):
                active_animal[i, j] = 1

    hr, fa, acc = np.nanmean(stat_array, 0).T
    hr_sem, fa_sem, acc_sem = (np.nanstd(stat_array, 0) / np.sqrt(np.tile(active_animal.sum(0).reshape(-1, 1), 3))).T

    # output shape: num_sessions x tone_type, e.g. 22x4 (22 sessions by 4 tone types (including target tone))
    
    return hr, fa, acc, hr_sem, fa_sem, acc_sem



#### plotting ####

