## Blip Image Captioning with GPT-2 Happy Model

This repository provides an implementation that combines the Blip Image Captioning model with a fine-tuned GPT-2 model for generating happy responses to the derived captions. The process involves utilizing the Salesforce Blip Image Captioning model to generate image captions and then passing those captions through the trained GPT-2 Happy Model.

### Requirements

To run the code in this repository, the following dependencies are required:

- Python 3.x
- Transformers library
- pandas
- numpy
- tqdm
- PIL (Python Imaging Library)

Please ensure that these dependencies are installed before proceeding.

### Preprocessing

The provided code includes preprocessing steps to prepare the data for training the GPT-2 Happy Model. It includes the following steps:

1. **Cleaning the Data**: The `clean` function reads a CSV file (`cleaned_hm.csv`) containing the happy moments data. It removes any rows with missing values in the 'cleaned_hm' column and drops duplicate entries. The cleaned data is saved as `happy_moments.csv`.

2. **Splitting the Data**: The `split` function reads the cleaned data from `happy_moments.csv` and splits it into training and test datasets. The data is randomly sampled, with 70% used for training and the remaining 30% for testing. The train and test datasets are saved as `train.csv` and `test.csv`, respectively.

3. **Tokenizing and Saving the Data**: The `tokenize_save` function tokenizes the text data in the train and test datasets using the GPT-2 tokenizer. The tokenized data is saved as `train_mod.txt` and `test_mod.txt`.

### Training the GPT-2 Happy Model

To train the GPT-2 Happy Model, follow these steps:

1. Load the necessary dependencies, including the `AutoTokenizer`, `AutoModelWithLMHead`, `Trainer`, `TrainingArguments`, `DataCollatorForLanguageModeling`, and `TextDataset` from the Transformers library.

2. Define the `load_dataset` function, which loads the tokenized train and test datasets using the GPT-2 tokenizer. It returns the train dataset, test dataset, and data collator required for language modeling.

3. Define the `train_setup` function, which sets up the training configuration for the GPT-2 model. It initializes the GPT-2 model, loads the train and test datasets, and specifies the training arguments such as the output directory, batch sizes, logging steps, and more. It returns the trainer object.

4. Instantiate the trainer by calling the `train_setup` function.

5. Execute the training process by running `trainer.train()`. The model will be trained for the specified number of epochs using the provided training arguments.

### Generating Happy Responses

After training the GPT-2 Happy Model, you can generate happy responses to derived captions using the Blip Image Captioning model. Here's how it works:

1. Load the necessary dependencies, including the `BlipProcessor`, `BlipForConditionalGeneration`, `AutoTokenizer`, and `AutoModelWithLMHead`.

2. Instantiate the `BlipProcessor` and `BlipForConditionalGeneration` models using the pre-trained weights from the "Salesforce/blip-image-captioning-large" checkpoint.

3. Specify the URL of the image for which you want to generate captions.

4. Open and convert the image from the URL to the RGB format using `PIL.Image`.

5. Perform conditional image captioning by providing a partial caption and tokenizing the image and text using the `BlipProcessor`. Generate the caption using the `BlipForConditionalGeneration` model's `

generate` method.

6. Pass the generated caption through the trained GPT-2 Happy Model to obtain a happy response. This involves tokenizing the caption using the GPT-2 tokenizer and using the GPT-2 model to generate the response.

By combining the Blip Image Captioning model and the GPT-2 Happy Model, you can generate happy responses based on derived captions from images. This repository provides the necessary code and preprocessing steps to perform this task.

## References

- J. Li, D. Li, C. Xiong, and S. Hoi, ‘BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation’. arXiv, 2022.
