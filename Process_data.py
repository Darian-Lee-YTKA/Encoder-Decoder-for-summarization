import torch
from torch import nn
import pandas as pd
import numpy as np
import re
import os

class Text_processer():
    def __init__(self):
        self.final_df = pd.DataFrame()
        self.chunk_df = None
        self.processed_x = None
        self.processed_y = None
        self.final_df = None

    def process_file(self, filename):
        """
        Goes through the entire processing pipeline
        """
        print("starting process file")

        with open(filename, 'r', encoding='utf-8') as f:
            chunks = f.read().split('\n')
        self.chunk_df = pd.DataFrame({'chunks': chunks})
        self.__whole_word_tokenize()

    def __whole_word_tokenize(self): # assuming we already have chunks
        # tokenizes based on full words
        print("starting whole_word_tokenize")
        self.chunk_df["chunks"] = self.chunk_df["chunks"].apply(lambda x: x.split(" "))
        self.__train_processing()



    def __train_processing(self):
        """
        Shrinks window size, removes stop words, lowercases text, removes dashes.
        """
        print("starting train processing")


        train = input("would you like to do train processing? y/n: ") # we are asking cause we likely wont do this for the target
        while train not in ["y", "n"]:
            train = input("Invalid input. would you like to do train processing? y/n: ")
        if train == "y":
            print("ok, processing starting! 😈")

            # lowercases input
            print("lowercasing input")
            self.chunk_df["chunks"] = self.chunk_df["chunks"].apply(
                lambda x: [word.lower() for word in x]
            )


            # remove all dashes
            print("removing dashes")
            self.chunk_df["chunks"] = self.chunk_df["chunks"].apply(
                lambda x: [re.sub(r"-+", "", word) for word in x] if isinstance(x, list) else x
            )

            # gets rid of stop words  (fast and sloppy method)
            print("removing stop words")
            stop_words = {"the", "is", "in", "and", "to", "of", "a", "an", "that", "this", "it", "for", "on", "with",
                          "as", "at", "by"}
            self.chunk_df["chunks"] = self.chunk_df["chunks"].apply(
                lambda x: [re.sub(r"-+", "", word) for word in x if word.lower() not in stop_words] if isinstance(x,
                                                                                                                  list) else x
            )

            # shrinks window size
            print("shrinking window size")
            self.chunk_df["chunks"] = self.chunk_df["chunks"].apply(
                lambda x: x[:150] + x[-150:] if len(x) > 300 else x
            ) # get the first and last 400
            self.processed_x = self.chunk_df
        else:
            # this means we are dealing with test. No other processing is needed
            self.processed_y = self.chunk_df

    def make_and_save_final_df(self):
        print("starting make and save final df")
        if self.processed_x is not None and self.processed_y is not None:
            # renaming columns from "chunks" to "src" for processed_x and "tgt" for processed_y
            self.processed_x = self.processed_x.rename(columns={'chunks': 'src'})
            self.processed_y = self.processed_y.rename(columns={'chunks': 'tgt'})

            # merging processed_x and processed_y into a single DataFrame
            self.final_df = pd.concat([self.processed_x, self.processed_y], axis=1)


            # saving the final DataFrame
            os.makedirs('processed_data', exist_ok=True)


            self.final_df.to_csv('processed_data/final_df_test.csv', index=False)
            print("Final DataFrame saved as 'processed_data/final_df_test.csv'.")
            




x_file = "data/test.txt.src"
y_file = "data/test.txt.tgt"


text_processer = Text_processer()
text_processer.process_file(x_file)
text_processer.process_file(y_file)
text_processer.make_and_save_final_df()











