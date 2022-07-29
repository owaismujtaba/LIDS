import os
import pdb

EPOCHS = 1
BATCH_SIZE = 1024
PATH = os.getcwd() + '/Data/'
n_rows = 15000

if __name__ == '__main__':

    print("1. Preprocessing Rajkumar2021 Paper")
    print("2. Test Rajkumar2021 Paper")
    print("3. Preprocessing Abdullah 2021 Paper")
    print("4. Test Abdullah 2021 Paper")
    print("5. Preprocessing Devrim 2022 Paper")
    print("6. Test Devrim 2022 Paper")
    print("7. Proposed Preprocessing: Create Dataset")
    print("8. Create PCA Dataset using processed dataset")
    print("9. Visualization")
    print("10. Proposed Model Binary Classification")
    print("11. Proposed Model Multi Classification")
    print("12. Plot Training and Validation")

    selection = int(input("Enter your implementation checker: "))

    if selection == 1:
        from RajKumar2021.data_utils import clean_dataset
        clean_dataset(PATH, n_rows)

    elif selection == 2:
        from RajKumar2021.model import test_model
        test_model(PATH, n_rows)

    elif selection == 3:
        from Abdullah2021.data_utils import clean_dataset
        clean_dataset(PATH, n_rows)

    elif selection == 4:
        from Abdullah2021.model import test_model
        test_model(PATH, EPOCHS, BATCH_SIZE, n_rows)

    elif selection == 5:
        from Devrim2022.data_utils import clean_dataset
        clean_dataset(PATH, n_rows)

    elif selection == 6:
        from Devrim2022.model import test_model
        test_model(PATH, EPOCHS, BATCH_SIZE, n_rows)

    elif selection == 7:
        from Proposed.data_utils import make_datasets
        make_datasets(PATH, n_rows)

    elif selection == 8:
        from Proposed.data_utils import create_pca_dataset
        create_pca_dataset()

    elif selection == 9:
        pdb.set_trace()
        from Proposed.data_utils import load_pca_dataset
        from vis_utils import plot_files_preprocessing, class_distributions
        from Proposed.pca_analysis import principal_component_varience, plot_pca_analysis
        print("1. Plotting Files Size Preprocessing")
        print("2. Plotting Preprocesses Class Distribution")
        print("3. Plotting Principal Component Variance")
        print("4. Plotting PCA Analysis Plot")

        dataset = load_pca_dataset()
        plot_files_preprocessing()
        class_distributions(dataset)
        plot_pca_analysis(dataset)
        principal_component_varience()

    elif selection == 10:
        from Proposed.train import trainer
        trainer(EPOCHS, BATCH_SIZE)

    elif selection == 11:
        from Proposed.train import trainer_multi
        trainer_multi(EPOCHS, BATCH_SIZE)

    elif selection == 12:
        from vis_utils import plot_proposed_model_accuracy_loss
        plot_proposed_model_accuracy_loss()
