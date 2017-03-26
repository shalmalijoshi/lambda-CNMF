Instructions for running lambda/s-CNMF:

    1. Data Preprocessing and creating data pickle:
        a. Extract clinical notes (except discharge summaries and echos) for the first 48 hours of adult (>=16 yrs) patient stay from MIMIC-III (See instructions to access MIMIC-III here: https://mimic.physionet.org/)

        b. Extract comorbidities associated with the same period of stay using: https://github.com/MIT-LCP/mimic-code/blob/master/concepts/comorbidity/elixhauser-ahrq-v37-with-drg.sql

        c. Extract 30-day mortality status associated with the patients.

        d. Use a custom clinical vocabulary (based on level-0 terms provided by UMLS) and SNOMED-CT to create a bag-of-concepts representation of the clinical notes. This can be done using preprocessing tools provided courtesy of Yacine Jernite and David Sontag at: https://github.com/clinicalml/ML-tools/tree/master/Preprocessing/ConceptCRFMatch

        e. Filter out clinical concepts with document frequencies >0.85 and <0.0005.

        f. In addition make sure all patients suffer from at least one chronic condition according to extracted comorbidity flags and filter out comorbidities if they have no associated patients recorded during the period of interest.

	g. Save the data in pickle file in the following  format:
           dictobj = {'Xtrain':Xtrain, 'Xtest':Xtest,
		      'comorbidities_first_train':ctrain, 'comorbidities_first_test':ctest,
		      'mortality_train':mortality_train,'mortality_test': mortality_test, 
		      'Ainit':Ainit
		     }

          where:
	  --> Xtrain: EHR data of patients in training data; #train_patients x #features numpy array
          --> Xtest: EHR data of patients in test data; #test_patients x #features numpy array
          --> ctrain: Binary (0/1) matrix representing comorbidities of patients in training data at the end of first 48 hours; #train_patients x #comorbidities bumpy array
          --> ctest: Same as above for test data
	  --> Ainit: Initialization for A; #train_patients x #comorbidities numpy array
	  --> mortality_train: numpy array of dimension #train_patients, mortality_train[j] = 1 if 30-day mortality recorded for patient j, 0 o.w.
          --> mortality_test: Same as above for test data
          
        h. Additional required files: filtered_conditions.txt (list of conditions) and filtered_vocabulary.txt (vocabulary list)

    2. Running s-CNMF and the baseline NMF+support (without simplex constraints):
        a. Let dataobj be saved as <data_folder_path>/EstimatorInput_strat_mortality<cv>.pkl
        b. main_constrainednmf_folds.py is the wrapper file that runs s-CNMF and NMF.
        c. The file takes the following parameters:
           -s <simplex constraint values> (s=0 corresponds to NMF s>0 to s-CNMF)
           -t <tau> (corresponding to tau in the paper, default 0.01)
           -l <loss> (sparse_gaussian or sparse_poisson, use sparse_poisson for count data)
           -i <run_ids> (cross-validation fold) 
           -b <bias 0/1> (weather or not to use a bias factor)
           -p <parallel optimization:0/1> (the default option 0 is recommended)
           -f data_folder_path
        d. A sample script is provided for reference: run.sh
	
    3. Baselines:
        a. MLC: Main code provided in main_logistic_regression_folds.py. To run follow:
           >> python main_logistic_regression_folds.py -s <comma separated regularization settings> -b <bias/default:1> -f <data_folder_path>
        b. LLDA: Export the data into Mallet compatible format. Follow installation and instructions to run LLDA here:
           http://mallet.cs.umass.edu/topics.php and http://www.mimno.org/articles/labelsandpatterns/

    4. Plotting phenotypes:
        a. See visualize_phenotypes.ipynb to visualize the top 25 terms associated with the phenotypes learned as columns of factor matrix A.
        b. Required files: 
           --> filtered_conditions.txt corresponding to the list of target conditions.
           --> filtered_vocabulary.txt corresponding to the custom vocabulary.
