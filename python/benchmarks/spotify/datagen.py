#!/usr/bin/env python3

import csv

def main():
    with open("spotify_dataset.csv", newline='') as fobj:
        reader = csv.reader(fobj)
        print(next(reader))
        for user_id, artist, track, playlist in reader:
            print(user_id, artist, track, playlist)
            break
        print(next(reader))
    pass

if __name__ == "__main__":
    main()
