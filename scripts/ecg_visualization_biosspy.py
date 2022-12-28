"""
:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
"""

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os

MAJOR_LW = 2.5
MINOR_LW = 1.5
MAX_ROWS = 10


def generate_ecg_plots(
    ts=None,
    raw=None,
    filtered=None,
    rpeaks=None,
    templates_ts=None,
    templates=None,
    heart_rate_ts=None,
    heart_rate=None,
    diagnostic=None,
    lead=None,
    save_plot=False,
):
    """Create a summary plot from the output of signals.ecg.ecg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw ECG signal.
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    """

    fig_raw, axs_raw = plt.subplots(3, 1, sharex=True)
    fig_raw.suptitle(f"Summary of {diagnostic} {lead}")

    # raw signal plot (1)
    axs_raw[0].plot(ts, raw, linewidth=MAJOR_LW, label="Raw", color="C0")
    axs_raw[0].set_ylabel("Amplitude (mV)")
    axs_raw[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs_raw[0].grid()

    # filtered signal with R-Peaks (2)
    axs_raw[1].plot(ts, filtered, linewidth=MAJOR_LW, label="Filtered", color="C0")

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    # adding the R-Peaks
    axs_raw[1].vlines(
        ts[rpeaks], ymin, ymax, color="m", linewidth=MINOR_LW, label="R-peaks"
    )

    axs_raw[1].set_ylabel("Amplitude (mv)")
    axs_raw[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs_raw[1].grid()

    # heart rate (3)
    axs_raw[2].plot(heart_rate_ts, heart_rate, linewidth=MAJOR_LW, label="Heart Rate")
    axs_raw[2].set_xlabel("Time (s)")
    axs_raw[2].set_ylabel("Heart Rate (bpm)")
    axs_raw[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs_raw[2].grid()

    fig = fig_raw

    fig_2 = plt.Figure()
    gs = gridspec.GridSpec(6, 1)

    axs_2 = fig_2.add_subplot(gs[:, 0])

    axs_2.plot(templates_ts, templates.T, "m", linewidth=MINOR_LW, alpha=0.7)
    axs_2.set_xlabel("Time (s)")
    axs_2.set_ylabel("Amplitude (mV)")
    axs_2.set_title(f"Template of {diagnostic} {lead}")
    axs_2.grid()

    output_directory = f"ecg_visualizations/{diagnostic}/{lead}/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    summary_path = os.path.join(
        output_directory, f"{diagnostic} {lead} summary.png"
    )
    template_path = os.path.join(
        output_directory, f"{diagnostic} {lead} template.png"
    )

    summary_file_name, summary_extension = os.path.splitext(summary_path)
    template_file_name, template_extension = os.path.splitext(template_path)
    counter = 1

    while os.path.exists(summary_path) and os.path.exists(template_path):
        summary_path = (
            summary_file_name + " (" + str(counter) + ")" + summary_extension
        )
        template_path = (
            template_file_name + " (" + str(counter) + ")" + template_extension
        )
        counter += 1

    # save to file
    if save_plot:
        fig.savefig(summary_path, dpi=200, bbox_inches="tight")
        fig_2.savefig(template_path, dpi=200, bbox_inches="tight")
