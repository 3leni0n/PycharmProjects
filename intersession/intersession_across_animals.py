import time
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from my_fun.my_fun import slack_spam


########################################################################################################################


def intersession_across_animals(experiment='2AFC_3', send_slack=False):

    # Register time
    time_start_total = time.time()

    ####################################################################################################################

    # Select folder where input data is
    folder_in = '/home/alexis/PycharmProjects/intersession/' + experiment + '/'

    # Select the folder where to save the PDF or create it if it doesn't exist
    folder_out = '/home/alexis/Documentos/intersession reports/' + experiment
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
    os.chdir(folder_out)

    ####################################################################################################################

    animals = os.listdir(folder_in)
    animals = [animals for animals in animals if animals.endswith('.csv')]
    animals.sort()
    # Remove animals from 2AFC_2 (batch 2) that never learnt the task to get a cleaner plot
    # animals.remove('328_intersession.csv')
    # animals.remove('326_intersession.csv')
    # animals.remove('331_intersession.csv')
    # animals.remove('334_intersession.csv')

    ####################################################################################################################

    # Plotting style
    ms = 3  # (mpl default for plt.plot=6 and for plt.scatter=6**2)
    lw = 1.5  # mpl default

    ####################################################################################################################

    with PdfPages('all_intersessions') as pdf:

        # PAGE 1

        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches portrait
        # fig = plt.figure(figsize=(11.69, 8.27))  # A4 size in inches landscape
        # fig = plt.figure()

        ################################################################################################################


        # SUMMARY TEXT

        # s1 = ('Dates: ' + df.Date.unique()[0] + ' - ' + df.Date.unique()[-1] + ', ' +
        #       'Subject: ' + df.Subject.unique()[0].astype(str) + ', ' +
        #       'Box: ' + df.Board.mode()[0][4] + ', ' +
        #       'Days: ' + str(n_dates) +
        #       '\n')

        ################################################################################################################

        # Create subplot's axes. Need to be outside for loop otherwise it overwrites the psychometric_curves
        ax = plt.subplot2grid((6, 1), (0, 0), rowspan=1, colspan=1)
        # ax1 = plt.subplot2grid((8, 1), (1, 0), rowspan=1, colspan=1)
        ax2 = plt.subplot2grid((6, 1), (1, 0), rowspan=2, colspan=1)
        ax3 = plt.subplot2grid((6, 1), (3, 0), rowspan=1, colspan=1)
        ax4 = plt.subplot2grid((6, 1), (4, 0), rowspan=1, colspan=1)
        ax5 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1)
        # ax6 = plt.subplot2grid((8, 1), (6, 0), rowspan=1, colspan=1)
        # ax7 = plt.subplot2grid((8, 1), (7, 0), rowspan=1, colspan=1)
        # axX = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)  # Regular plot (1 row x 1 column grid)

        for i in range(len(animals)):

            df = pd.read_csv(folder_in + animals[i])  # Load intersession data from an animal
            df_dt = pd.to_datetime(df.Dates)  # Convert to dates to datetime, x axis

            ################################################################################################################

            # PLOT 0: RESPONSES

            time_start = time.time()

            # ax = plt.subplot2grid((8, 1), (0, 0), rowspan=1, colspan=1)
            # ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

            ax.plot(df_dt, df.Responses, marker='o', ms=ms, lw=lw, label=df.Subject.unique()[0])

            if i == len(animals) - 1:

                # ax.set_xticklabels([ax.get_xticklabels()], rotation=45, ha='center')
                ax.set_xticklabels([])
                ax.set_ylabel('Responses')
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                # Instantiate a second axes that shares the same x-axis
                ax_twin = ax.twinx()
                ax_twin.set_ylim(ax.get_ylim())
                # ax_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
                ax_twin.set_yticklabels([])
                ax_twin.spines['top'].set_visible(False)
                ax_twin.spines['bottom'].set_visible(False)

            ################################################################################################################

            # # PLOT 1: WATER
            #
            # time_start = time.time()
            #
            # # ax1 = plt.subplot2grid((8, 1), (1, 0), rowspan=1, colspan=1)
            # # ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
            #
            # ax1.plot(df_dt, df.Water, marker='o', ms=ms, lw=lw)
            #
            # if i == len(animals) - 1:
            #
            #     # ax1.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
            #     ax1.set_xticklabels([])
            #     ax1.set_ylabel('Water')
            #     ax1.spines['top'].set_visible(False)
            #     ax1.spines['bottom'].set_visible(False)
            #
            #     # Instantiate a second axes that shares the same x-axis
            #     ax1_twin = ax1.twinx()
            #     ax1_twin.set_ylim(ax1.get_ylim())
            #     # ax1_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
            #     ax1_twin.set_yticklabels([])
            #     ax1_twin.spines['top'].set_visible(False)
            #     ax1_twin.spines['bottom'].set_visible(False)

            ################################################################################################################

            # PLOT 2: ACCURACY FOR MAXIMUM EVIDENCES

            time_start = time.time()

            # ax2 = plt.subplot2grid((8, 1), (2, 0), rowspan=1, colspan=1)
            # ax2 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

            ax2.plot(df_dt, df.AccMaxEvi, marker='o', ms=ms, lw=lw)

            # Plot horizontal lines
            ax2.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
            ax2.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
            ax2.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

            if i == len(animals) - 1:

                # ax2.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
                ax2.set_xticklabels([])
                ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax2.set_ylabel('Acc. Max. Evi.\n(%)')
                ax2.spines['top'].set_visible(False)
                ax2.spines['bottom'].set_visible(False)

                # Instantiate a second axes that shares the same x-axis
                ax2_twin = ax2.twinx()
                ax2_twin.set_ylim(ax2.get_ylim())
                # ax2_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
                ax2_twin.set_yticklabels([])
                ax2_twin.spines['top'].set_visible(False)
                ax2_twin.spines['bottom'].set_visible(False)

            ################################################################################################################

            # PLOT 3: REPEATING BIAS

            time_start = time.time()

            # ax3 = plt.subplot2grid((8, 1), (3, 0), rowspan=1, colspan=1)
            # ax3 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

            ax3.plot(df_dt, df.RepBias, marker='o', ms=ms, lw=lw)

            if i == len(animals) - 1:

                # ax3.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
                ax3.set_xticklabels([])
                ax3.set_ylabel('Rep. Bias')
                ax3.spines['top'].set_visible(False)
                ax3.spines['bottom'].set_visible(False)

                # Instantiate a second axes that shares the same x-axis
                ax3_twin = ax3.twinx()
                ax3_twin.set_ylim(ax3.get_ylim())
                # ax3_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
                ax3_twin.set_yticklabels([])
                ax3_twin.spines['top'].set_visible(False)
                ax3_twin.spines['bottom'].set_visible(False)

            ################################################################################################################

            # PLOT 4: MISS RATE

            time_start = time.time()

            # ax4 = plt.subplot2grid((8, 1), (4, 0), rowspan=1, colspan=1)
            # ax4 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

            ax4.plot(df_dt, df.MissRate, marker='o', ms=ms, lw=lw)

            # Plot horizontal lines
            ax4.axhline(0.5, color='tab:gray', linestyle='--')  # Chance level
            ax4.axhline(0.25, color='tab:gray', linestyle=':')  # Accuracy 0.25
            ax4.axhline(0.75, color='tab:gray', linestyle=':')  # Accuracy 0.75

            if i == len(animals) - 1:
                # ax4.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
                ax4.set_xticklabels([])
                ax4.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax4.set_ylabel('Misses\n(%)')
                ax4.spines['top'].set_visible(False)
                ax4.spines['bottom'].set_visible(False)

                # Instantiate a second axes that shares the same x-axis
                ax4_twin = ax4.twinx()
                ax4_twin.set_ylim(ax4.get_ylim())
                # ax4_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
                ax4_twin.set_yticklabels([])
                ax4_twin.spines['top'].set_visible(False)
                ax4_twin.spines['bottom'].set_visible(False)

            ################################################################################################################

            # PLOT 5: STAGES/SUBSTAGES/MOTOR

            time_start = time.time()

            # ax5 = plt.subplot2grid((8, 1), (5, 0), rowspan=1, colspan=1)
            # ax5 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

            ax5.plot(df_dt, df.Stage, marker='o', ms=ms, lw=lw)

            if i == len(animals) - 1:
                # ax5.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
                # ax5.set_xticklabels([])
                ax5.set_ylabel('Stage')
                ax5.spines['top'].set_visible(False)
                ax5.spines['bottom'].set_visible(False)

                # Instantiate a second axes that shares the same x-axis
                ax5_twin = ax5.twinx()
                ax5_twin.set_ylim(ax5.get_ylim())
                # ax5_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
                ax5_twin.set_yticklabels([])
                ax5_twin.spines['top'].set_visible(False)
                # ax5_twin.spines['bottom'].set_visible(False)

            ################################################################################################################

            # # PLOT 6: SOUND CHECKS
            #
            # time_start = time.time()
            #
            # # ax6 = plt.subplot2grid((8, 1), (6, 0), rowspan=1, colspan=1)
            # # ax6 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
            #
            # ax6.plot(df_dt, df.NoSound, marker='o', ms=ms, lw=lw)
            #
            # if i == len(animals) - 1:
            #     # ax6.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
            #     # ax6.set_xticklabels([])
            #     ax6.set_ylabel('NoSound')
            #     ax6.spines['top'].set_visible(False)
            #     # ax6.spines['bottom'].set_visible(False)
            #
            #     # Instantiate a second axes that shares the same x-axis
            #     ax6_twin = ax6.twinx()
            #     ax6_twin.set_ylim(ax6.get_ylim())
            #     # ax6_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
            #     ax6_twin.set_yticklabels([])
            #     ax6_twin.spines['top'].set_visible(False)
            #     # ax6_twin.spines['bottom'].set_visible(False)

            ################################################################################################################

            # # PLOT 7: PROBABILITIES DIFFICULT TRIALS
            #
            # time_start = time.time()
            #
            # # ax7 = plt.subplot2grid((8, 1), (7, 0), rowspan=1, colspan=1)
            # # ax7 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
            #
            # ax7.plot(df_dt, df.P, marker='o', ms=ms, lw=lw)
            #
            # if i == len(animals) - 1:
            #     ax7.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')
            #     ax7.set_xticklabels([])
            #     ax7.set_ylabel('P')
            #     ax7.spines['top'].set_visible(False)
            #     ax7.spines['right'].set_visible(False)
            #
            #     # Instantiate a second axes that shares the same x-axis
            #     ax7_twin = ax7.twinx()
            #     ax7_twin.set_ylim(ax7.get_ylim())
            #     # ax7_twin.set_yticks(list(np.arange(0, trials.max() + 100, 100)[0::2]))
            #     ax7_twin.set_yticklabels([])
            #     ax7_twin.spines['top'].set_visible(False)
            #     ax7_twin.spines['right'].set_visible(False)

        ax.legend(loc='lower right', fontsize='xx-small', frameon=True)
        plt.draw()  # The tick label strings are not populated until a draw method has been called
        # (https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.axes.Axes.get_xticklabels.html)
        fig.canvas.draw()  # Alternative
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
        ax5.set_xlabel('Date')

        ################################################################################################################

        time_start_savepag1 = time.time()
        pdf.savefig()  # saves the current figure into a pdf page
        time_end_savepag1 = time.time()
        runtime_savepag1 = time_end_savepag1 - time_start_savepag1
        print("'Saving 1st page in pdf' took", round(runtime_savepag1, 2), 'seconds to run')

        # plt.savefig(setup + 'all_intersessions.png')  # Save as png as well
        plt.close()

        ################################################################################################################

        print(time.time())
        time.sleep(60)
        print(time.time())

        # This block needs to be the last otherwise it sends the file too soon and corrupted
        if send_slack:
            with open('/home/alexis/slack_bot_token', 'r') as f:  # Get slack bot token
                slack_bot_token = f.read().replace('\n', '')

            os.environ['SLACK_BOT_TOKEN'] = slack_bot_token
            # filepath = folder_pdf_out + '/' + df.Session.unique()[0]
            filepath = folder_out + '/' + 'all_intersessions'
            slack_spam(msg='Hey buddy!', filepath=filepath, userid='#pv_nmdar_eranet_reports')  # Alexis: 'U01DDHH7LLX'