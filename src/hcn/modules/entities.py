from enum import Enum
import numpy as np


class EntityTracker():

    def __init__(self):
        self.entities = {
                '<cuisine>' : None,
                '<location>' : None,
                '<party_size>' : None,
                '<rest_type>' : None,
                }
        self.num_features = 4 # tracking 4 entities
        self.rating = None

        # constants
        self.party_sizes = ['1', '2', '3', '4', '5', '6', '7', '8', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
        self.locations = ['bangkok', 'beijing', 'bombay', 'hanoi', 'paris', 'rome', 'london', 'madrid', 'seoul', 'tokyo'] 
        self.cuisines = ['british', 'cantonese', 'french', 'indian', 'italian', 'japanese', 'korean', 'spanish', 'thai', 'vietnamese']
        self.rest_types = ['cheap', 'expensive', 'moderate']

        self.EntType = Enum('Entity Type', '<party_size> <location> <cuisine> <rest_type> <non_ent>')


    def ent_type(self, ent):
        if ent in self.party_sizes:
            return self.EntType['<party_size>'].name
        elif ent in self.locations:
            return self.EntType['<location>'].name
        elif ent in self.cuisines:
            return self.EntType['<cuisine>'].name
        elif ent in self.rest_types:
            return self.EntType['<rest_type>'].name
        else:
            return ent


    def extract_entities(self, utterance, update=True):
        tokenized = []
        for word in utterance.split(' '):
            entity = self.ent_type(word)
            if word != entity and update:
                self.entities[entity] = word

            tokenized.append(entity)

        return ' '.join(tokenized)


    def context_features(self):
       keys = list(set(self.entities.keys()))
       self.ctxt_features = np.array( [bool(self.entities[key]) for key in keys], 
                                   dtype=np.float32 )
       return self.ctxt_features


    def action_mask(self):
        print('Not yet implemented. Need a list of action templates!')
