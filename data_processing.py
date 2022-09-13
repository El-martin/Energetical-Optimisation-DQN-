""" Tools to extract data from the Meteonet database
     or to smooth data before plotting it """
import numpy as np
import os
import pickle
os.chdir('C:/Users/Moi/Desktop/')


def date_to_day(date):
    """ Turns a full date into the number of the day"""
    month = int(date[4:6])
    day = int(date[6:8])
    return 31 * (month - 1) + day


def time_to_minutes(date):
    """ Turns a full date into the number of minutes elapsed since last midnight"""
    h = int(date[9:11])
    m = int(date[12:14])
    return 60*h + m


def extract_weather_data(filename_in, filename_out):
    """ Specific to the Meteonet database """
    donnees = []

    with open(filename_in, 'r') as fichier:
        ligne = fichier.readline()
        for i in range(2*29416496):
            ligne = fichier.readline().split(',')
            if ligne != ['\n'] and ligne[0] == '14066001':  # '1027003' SE '14066001' NW :
                date = ligne[4]
                try:                    # [day num, minute, hum, Temp]
                    entree = [date_to_day(date), time_to_minutes(date), float(ligne[-4])*.5,
                              float(ligne[-2])-273.15+8]
                except ValueError:
                    print(f"encountered issue with the following data : h: {ligne[-4]}, T: {ligne[-2]}")
                if '' not in entree:
                    donnees.append(entree)

            if i % 10e5 == 0:
                print(i/(2*294164.96), '%')

    donnees = np.array(donnees)

    with open(filename_out, 'wb') as my_pickler:
        pickle.dump(donnees, my_pickler)
    print(f"Saved {filename_out} successfully")


def smooth_curve(data, factor):
    """ Smoothing the curve using the mean of a given number of successive values (factor)"""
    output_data = []
    for i in range(len(data[factor - 1:])):
        batch = data[i:i + factor]  # data is split into successive batches to compute a mean value
        mean = sum(batch) / factor  # computing this mean
        output_data.append(mean)  # adding it to the 'smoothed' list
    return output_data
