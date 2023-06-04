import warnings

warnings.filterwarnings('ignore')

import train
import preprocess

def main():
    preprocess.main()
    train.main()
    
if __name__ == "__main__":
    main()

