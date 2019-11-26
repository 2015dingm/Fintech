#%%
import os
import shutil

# files in each folder
file_list = ['main.py', 'deeplearning_stock.sh',
             'single_run.py', 'tuning.py']

## NOTE!!! change path here
old_folder = '/project/polson/GavinFeng/mingding/TSM/1006/other_methods_cs/code/'

#%%
# create folder
for i in range(1, 56):
    folder = os.path.dirname(os.path.dirname(old_folder))+'/asset{}/'.format(i)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # copy file in file_list
    for file in file_list:
        shutil.copyfile(old_folder + file, folder + file)
    ############################################################################
    # generate 'script{test_year}.py' file for each folder for each year'
    os.chdir(folder)
    insert_line = 30 # which line do you insert parameters in main.py

    start_year = 2006
    end_year = 2016
    # write py file
    for test_year in range(start_year, end_year):
        line_count = 1
        f0 = open('main.py', 'r', encoding='utf-8')
        f = open('script' + str(test_year) + '.py', 'w', encoding='utf-8')
        while line_count < insert_line:
            line_count = line_count + 1
            f.write(f0.readline())
        f.write('# change X and Y dataset\n')
        f.write('asset_i = {}\n'.format(i))
        f.write('test_year = ' + str(test_year) + '\n')
        f.write(f0.read())
        f0.close()
        f.close()
    ############################################################################
    # write sh file to submit 'script{test_year}.py'
    for test_year in range(start_year, end_year):
        f0 = open('deeplearning_stock.sh', 'r')
        f = open('submit'+str(test_year)+'.sh', 'w')
        f.write(f0.read())
        f.write('srun python3 script' + str(test_year) +'.py > output' + str(test_year) + '.txt 2>&1 \n')
        f0.close()
        f.close()

    # write a main submit sh file
    f = open('submit.sh', 'w')
    for test_year in range(start_year, end_year):
        f.write('sbatch --account=pi-ngp submit'+str(test_year)+'.sh\n')

    f.close()


####################################
# write sh file
os.chdir(old_folder)
# create submit.sh
f1 = open('../submit.sh', 'w')
for i in range(1, 56):
    folder = './asset{}/'.format(i)
    f1.write('cd {}\n'.format(folder))
    f1.write('sh submit.sh\n')
    f1.write('cd ..\n')
# write all and close
f1.close()




