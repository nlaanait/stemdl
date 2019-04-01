import os
import json
import numbers


class Saver(object):
    """Save loss value to JSON.
    
    Parameters
    ----------
    * `checkpoint_path` : str
        location where checkpoint will be saved to
        
    * `filename` : str
        Name of the file to save.
    """
    def __init__(self, checkpoint_path, filename):
        self.checkpoint_path = checkpoint_path
        self.filename = filename
        self.savefile = os.path.join(self.checkpoint_path, self.filename)

    def _convert_fields(self, result_field):
        """Convert numpy types to python objects.
        
        Necessary to serialize results.
        
        Parameters
        ----------
        * `result_field` : list
            Field consisting of numpy types to be converted.
        """
        converted_field = []
        for dim in result_field:
            if isinstance(dim, numbers.Integral):
                converted_field.append(int(dim))
            elif isinstance(dim, numbers.Real):
                converted_field.append(float(dim))
            else:
                converted_field.append(dim)

        return converted_field
    
    def load(self):
        """Load previous results"""
        with open(self.savefile, 'r') as infile:
            data = json.load(infile)
        return data 
            
    def save(self, res):
        """Save results to disk in JSON format.
        
        Parameters
        ----------
        * `res` : np.array
            The training loss at the end of each train session.
        """
        try:
            data = self.load()
            data['train_loss'].extend(self._convert_fields([res]))
        except FileNotFoundError:
            data = {}
            data['train_loss'] = self._convert_fields([res])

        with open(self.savefile, 'w') as outfile:
            json.dump(data, outfile)
