
import pymongo
from model import Pairwise_Sentence_Model, Target_Label_Detector_base_on_PSM



class Google_Map_Review_Detector:
    def __init__(self):
        # init database
        self.database_open_init()
        self.database_create_init()
        # detector
        self.window_size = 10 # it must be even
        self.model_init()


    def database_open_init(self):
        self.myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        self.database = self.myclient['Google_Map_Review_Database(Scaper)']
        self.restaurant_name_collection = self.database['restaurant_name']
    

    def database_create_init(self):
        self.database_of_predictor = self.myclient['Google_Map_Review_Database(Predictor)']
        self.restaurant_name_collection_of_predictor = self.database_of_predictor['restaurant_name']


    def new_collection_init(self, restaurant_name):
        # store prediction and other info for all relative restaurant
        restaurant_prediction_collection = self.database_of_predictor[restaurant_name]
        return restaurant_prediction_collection


    def model_init(self):
        self.tld_obj = Target_Label_Detector_base_on_PSM(total_target_label=['環境','服務'])


    def detect_review_semantic(self, review, target_label):
        if review is not None and target_label in review:    
            upper_index = review.index(target_label) + int(self.window_size/2)
            lower_index = review.index(target_label) - int(self.window_size/2)
            window_text = ''.join(review[lower_index : upper_index+len(target_label)-1+1])
            predicted_semantic = self.tld_obj.forward(target_label=target_label, source_sent=[window_text])
            return predicted_semantic, window_text
        elif review is None:
            return None, None
        else:
            return 0, None


    def forward(self):
        for element in self.restaurant_name_collection.find():
            restaurant_name = element['restaurant_name']
            restaurant_collection = self.database[restaurant_name]
            # store data to mongoDB - (restaurant name part)
            restaurant_name_data = [{'restaurant_name' : restaurant_name}]
            self.restaurant_name_collection_of_predictor.insert_many(restaurant_name_data)
            # insert collection name of prediction
            restaurant_prediction_collection = self.new_collection_init(restaurant_name)
            prediction_data = list()
            # review semantic prediction part
            for element_for_rc in restaurant_collection.find():
                review = element_for_rc['review']
                star = element_for_rc['star']
                date = element_for_rc['date']
                user_name = element_for_rc['user_name'] 
                # target label part
                for target_label in ['環境', '服務']:
                    data = {
                            'user_name' : user_name,
                            'review' : review,
                            'star' : star,
                            'date' : date,
                            'predict' : None,
                            'target_label' :  target_label,
                            'window_text' : None
                            }
                    predicted_semantic, window_text = \
                        self.detect_review_semantic(review=review, target_label=target_label)
                    # collect data to prediction_data of collection - (prediction part)
                    if predicted_semantic != 0 and predicted_semantic is not None:
                        data['predict'] = predicted_semantic
                        data['window_text'] = window_text
                        prediction_data.append(data)
            # store data to mongoDB - (prediction part)
            if len(prediction_data) != 0:
                restaurant_prediction_collection.insert_many(prediction_data)
            
                
                

if __name__ == '__main__':
    obj = Google_Map_Review_Detector()
    obj.forward()
