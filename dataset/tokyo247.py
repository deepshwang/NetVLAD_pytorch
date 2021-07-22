from dataset.tokyo import TokyoValTestDataset, TokyoTrainWholeDataset


class Tokyo247QueryDataset(TokyoValTestDataset):
    """
    Tokyo24/7 Query Dataset (query_v2)
    """
    def __init__(self, *args, **kwargs):
        super(Tokyo247QueryDataset, self).__init__(*args, **kwargs, dbStruct_path='/media/TrainDataset/tokyo247/tokyo247.mat', 
                                                                       dataset='tokyo247', 
                                                                       split='test', 
                                                                       db_q='247query_v2')
        self.utm = self.utmQ

class Tokyo247DBDataset(TokyoValTestDataset):
    """
    Tokyo24/7 Query Dataset (query_v2)
    """
    def __init__(self, *args, **kwargs):
        super(Tokyo247DBDataset, self).__init__(*args, **kwargs, dbStruct_path='/media/TrainDataset/tokyo247/tokyo247.mat', 
                                                                       dataset='tokyo247', 
                                                                       split='test', 
                                                                       db_q='database')
        self.utm = self.utmDb


class Tokyo247TrainWholeDataset(TokyoTrainWholeDataset):
    def __init__(self, *args, **kwargs):
        super(Tokyo247TrainWholeDataset, self).__init__(*args, **kwargs, dbStruct_path='/media/TrainDataset/tokyo247/tokyo247.mat', 
                                                                        dataset='tokyo247', 
                                                                        split='train', 
                                                                        db_q=None)