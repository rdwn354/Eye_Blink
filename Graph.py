import csv
import os

import matplotlib.pyplot as plt
import numpy as np

data = os.listdir("Record/backup")
backup_files_row = ", ".join(data)
print("here's for data : ")
print(backup_files_row, end="\n\n")

csv_file = input("Input name: ")
time = int(input("Input time: "))

path_csv = f'Record/backup/{csv_file}/{csv_file}{time}.csv'


class BlinkAnalyzer:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename

    def count(self, row_index):
        with open(self.csv_filename, 'r') as file:
            reader = csv.reader(file)
            for row_number, row in enumerate(reader):
                if row_number == row_index:
                    values = [float(value) for value in row if value != '0']
                    blink = len(values)
                    return values, blink
        return None, 0

    def visualize(self, minutes):
        blink = []
        value = []

        for i in range(minutes):
            row_index = i
            values, blinks = self.count(row_index)
            if values:
                blink.append(blinks)
                value.append(values)
                print(f"Non-zero values in row {row_index}: {values}")
            else:
                blink.append(blinks)
                value.append(values)
                print(f"Row {row_index} not found or contains only zeros.")


        print(blink)
        print(value)

        time_blink_1d = [time for sublist in value for time in sublist]

        return time_blink_1d, blink, value


class ShowGraph:
    def plot_data(self, analyzer, minutes):
        time_blink_1d, blink, value = analyzer.visualize(minutes)

        screen_width_px = 680
        screen_height_px = 300
        desired_dpi = 100  # You can adjust this value as needed
        screen_width_in = screen_width_px / desired_dpi
        screen_height_in = screen_height_px / desired_dpi

        fig = plt.figure(figsize=(screen_width_in, screen_height_in), dpi=desired_dpi)
        # creating tren
        x = np.arange(len(blink))
        y = np.array(blink)
        slope, intercept = np.polyfit(x, y, 1)

        # Menentukan arah tren
        if slope > 0:
            trend_direction = "up"
        elif slope < 0:
            trend_direction = "down"
        else:
            trend_direction = "flat"

        treadline = intercept + slope * x

        # Plot for blink counts
        plt.subplot(1, 2, 1)
        plt.plot(blink)
        plt.plot(treadline, label=f'Garis Tren {trend_direction}', color='red', linewidth = 1)
        plt.xlabel('Time (minute)')
        plt.ylabel('Blink')
        plt.title('Blink Count')

        lowest_value = min(blink)
        lowest_index = np.argmin(blink)

        highest_value = max(blink)
        highest_index = np.argmax(blink)

        median_blink = len(blink) // 2
        values_under_10 = len([x for x in blink if x <= 10])

        status = values_under_10 >= median_blink

        # Printing the indices (if needed)
        if lowest_index != -1:
            print("Lowest element index: ", lowest_index)
        if highest_index != -1:
            print("Highest element index: ", highest_index)

        plt.subplot(1, 2, 2)
        plt.text(0, 0.9, f'Name: {csv_file}', fontsize=8)
        plt.text(0, 0.8, f'Total time : {len(blink)} minutes', fontsize=8)
        plt.text(0, 0.7, f'Blink under set point : {values_under_10} times', fontsize=8)
        plt.text(0, 0.6, f'Lowest blink in minute {lowest_index} with {lowest_value} blinks', fontsize=8)
        plt.text(0, 0.5, f'Highest blink in minute {highest_index} with {highest_value} blinks', fontsize=8)
        plt.text(0, 0.4, f'Indicate : {"Computer Vision Syndrome" if trend_direction == "down" else "Normal"}', fontsize=8)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')

        plt.tight_layout()

        # Display the plot
        plt.show()


analyzer = BlinkAnalyzer(path_csv)

graph_display = ShowGraph()

# Call plot_data method to display the graph for the given time
graph_display.plot_data(analyzer, time)
