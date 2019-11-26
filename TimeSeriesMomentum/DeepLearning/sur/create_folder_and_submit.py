#%%
import os
import shutil

# files in each folder
file_list = ['main.py', 'deeplearning_stock.sh',
             'single_run.py', 'tuning.py', 'NN_function_ts.py', 'SelfDefinedRegularizations.py']

## NOTE!!! change path here
old_folder = '/project/polson/GavinFeng/mingding/1016_DL/sur/code/'

#%%
# create folder
start_year = 2006
end_year = 2016
for test_year in range(start_year, end_year):
    folder = os.path.dirname(os.path.dirname(old_folder))+'/test_year{}/'.format(test_year)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # copy file in file_list
    for file in file_list:
        shutil.copyfile(old_folder + file, folder + file)
    ############################################################################
    # generate 'script{test_year}.py' file for each folder for each year'
    os.chdir(folder)
    insert_line = 30 # which line do you insert parameters in main.py


    # write py file
    # for i in range(1, 56):
    line_count = 1
    f0 = open('main.py', 'r', encoding='utf-8')
    f = open('script.py', 'w', encoding='utf-8')
    while line_count < insert_line:
        line_count = line_count + 1
        f.write(f0.readline())
    f.write('# change X and Y dataset\n')
    # f.write('asset_i = {}\n'.format(i))
    f.write('test_year = ' + str(test_year) + '\n')
    f.write(f0.read())
    f0.close()
    f.close()
    ############################################################################
    # write sh file to submit 'script{test_year}.py'
    # for i in range(1, 56):
    f0 = open('deeplearning_stock.sh', 'r')
    f = open('submit1.sh', 'w')
    f.write(f0.read())
    f.write('srun python3 script.py > output.txt 2>&1 \n')
    f0.close()
    f.close()

    # write a main submit sh file
    f = open('submit.sh', 'w')
    f.write('sbatch --account=pi-ngp submit1.sh\n')

    f.close()

####################################
# write sh file
os.chdir(old_folder)
# create submit.sh
f1 = open('../submit.sh', 'w')
for test_year in range(start_year, end_year):
    folder = './test_year{}/'.format(test_year)
    f1.write('cd {}\n'.format(folder))
    f1.write('sh submit.sh\n')
    f1.write('cd ..\n')
# write all and close
f1.close()



