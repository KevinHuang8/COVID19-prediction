import numpy as np
import os

np.set_printoptions(precision=5)
import utils


def score_all_predictions(pred_file, date, model_date, mse=False, key='cases', bin_cutoffs=[20, 200, 1000]):
    true_data = utils.get_processed_df('nyt_us_counties_daily.csv')
    cum_data = utils.get_processed_df('nyt_us_counties.csv')
    proc_score_date = utils.process_date(date, true_data)
    proc_model_date = utils.process_date(model_date, true_data)
    raw_pred_data = np.genfromtxt(pred_file, delimiter=',', skip_header=1, dtype=np.str)
    date_preds = np.array([row for row in raw_pred_data if date in row[0]])
    all_fips = np.array([row[0].split('-')[-1] for row in date_preds])
    all_preds = date_preds[:, 1:].astype(np.float)
    true_data = np.array([utils.get_region_data(true_data, fips, proc_date=proc_score_date, key=key) for fips in all_fips])
    cum_data = np.array([utils.get_region_data(cum_data, fips, proc_date=proc_model_date, key=key) for fips in all_fips])
    return get_scores(all_fips, all_preds, true_data, cum_data, mse=mse, bin_cutoffs=bin_cutoffs)


def get_scores(all_fips, all_preds, true_data, cum_data, bin_cutoffs=[20, 1000], mse=False):
    tot_loss = 0
    bin_losses, bin_counts = np.zeros(len(bin_cutoffs) + 1), np.zeros(len(bin_cutoffs) + 1)
    losses = {}
    z = 0
    for fips, preds, true_number, cum_number in zip(all_fips, all_preds, true_data, cum_data):
        if mse:
            loss = (preds[4] - true_number) ** 2
            if fips == '36061':
                print('NYC: ', preds[4], true_number, loss)
            if true_number == 0:
                losses[fips] = 0
            else:
                losses[fips] = abs(preds[4] - true_number) / true_number
        else:
            loss = pinball_loss(preds, true_number)
        losses[fips] = loss
        if loss == 0:
            z += 1
        tot_loss += loss
        done = False
        for i, bc in enumerate(bin_cutoffs):
            if cum_number <= bc:
                bin_losses[i] += loss
                bin_counts[i] += 1
                done = True
                break
        if not done:
            bin_losses[-1] += loss
            bin_counts[-1] += 1

    return tot_loss / len(all_preds), bin_losses / bin_counts, losses


def pinball_loss(preds, true_val, p_vals=np.arange(0.1, 1.0, 0.1)):
    loss = 0
    for pred, p in zip(preds, p_vals):
        delta = np.abs(true_val - pred)
        if pred < true_val:
            loss += p * delta
        else:
            loss += (1 - p) * delta
    return loss / len(p_vals)


if __name__ == '__main__':
    from operator import itemgetter
    import datetime as dt
    from collections import defaultdict
    pred_file = '../../predictions/predictions-2020-05-12.csv'
    start_date = '2020-05-12'
    date = dt.datetime.strptime(start_date, '%Y-%m-%d')
    horizon = 30
    end_date = date + dt.timedelta(days=horizon)
    total_score = 0
    total_score_mse = 0
    losses = defaultdict(int)
    while date < end_date:
        date_str = date.strftime('%Y-%m-%d')
        tot_loss, _, losses_info = score_all_predictions(pred_file, date_str, date_str, key='deaths')
        total_score += tot_loss
        for fips in losses_info:
            losses[fips] += losses_info[fips]
        total_score_mse += score_all_predictions(pred_file, date_str, date_str, key='deaths', mse=True)[0]
        date = date + dt.timedelta(days=1)
    for fips in losses:
        losses[fips] /= horizon
    total_score /= horizon
    total_score_mse /= horizon
    print(total_score, total_score_mse)

    all_losses = list(losses.items())
    print(sorted(all_losses,key=itemgetter(1)))