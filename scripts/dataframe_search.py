import pandas as pd
import sys, os, subprocess, shlex

def get_file_paths(wdir, tag):
    f_list = [itm for itm in os.listdir(wdir) if tag in itm and 'pdf' in itm] 
    f_list = [os.path.join(wdir, itm) for itm in  f_list]
    print(f_list)
    return f_list

def combine_df(wdir, tag, delete=False):
    f_path = get_file_paths(wdir, tag)
    #group_id = [int(path.split('_')[3]) for path in f_path if 'params' in path]
    for path in f_path:
        print(path)
        if 'params' in path:
            master_params = pd.read_pickle(path)
        if 'results' in path:
            master_results = pd.read_pickle(path)
    for path in f_path:
        g_id = int(path.split('_')[3]) 
        print('group_id:{}'.format(g_id))
        pdf_group = pd.read_pickle(path)
        print('read: {}'.format(path))
        if 'params' in path:
            master_params['group_%d' % g_id] = pdf_group['group_%d' % g_id] 
        elif 'results' in path:
            master_results['group_%d' % g_id] = pdf_group['group_%d' % g_id]
            args = 'rm %s' % path
            args = shlex.split(args)
        if delete:
            try:
                subprocess.run(args, check=True, timeout=10)
            except subprocess.SubprocessError as e:
                print('could not delete file:{}'.format(path))

    master_results.to_csv(os.path.join(wdir,'results_{}.csv'.format(tag)))
    master_params.to_csv(os.path.join(wdir,'params_{}.csv'.format(tag)))
    master_results.to_pickle(os.path.join(wdir,'results_{}.pkl'.format(tag)))
    master_params.to_pickle(os.path.join(wdir,'params_{}.pkl'.format(tag)))
    return master_results, master_params

def extract_vals_params(df_results, df_params):
    pass

def main(wdir, tag):
    master_results, master_params= combine_df(wdir, tag)
    master_results
    return

if __name__ == "__main__":
    wdir, job_id = sys.argv[1:]
    main(wdir, job_id)



        

