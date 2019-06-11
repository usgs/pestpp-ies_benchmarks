# TODO: test variance and mean of draws, add chenoliver and test approx and full solution
import os
import shutil
import platform
import numpy as np
import pandas as pd
import platform
import matplotlib.pyplot as plt
import pyemu

tests = """0) 10par_xsec "standard user mode" - draw reals from par-bounds prior and obs noise from weights
0a) 10par_xsec same as 0) but with multple lambda 
1) 10par_xsec start with existing par csv and obs csv - using empirical parcov and obscov
1a) 10par_xsec start with existing par csv and obs csv - using parcov file
2) 10par_xsec start with existing par csv and drawing obs en from weights 
3) 10par_xsec restart with full simulated obs en
3a) 10par_xsec restart with failed runs in simulated obs en
3b) 10par_xsec restart with failed runs and bad phi runs in simulated obs en with multiple lam
4) 10par_xsec reg_factor = 0.5 test
5)  10par_xsec full solution test with standard draw mode
5a) 10par_xsec full solution test with empirical parcov
6) freyberg "standard user mode" - draw reals from par-bounds prior and obs noise from weights
6a) freyberg same as 0) but with multple lambda 
7) freyberg draw par en from full parcov supplied in file
8) freyberg full solution with empirical parcov - supplied par csv, obs csv and restart csv with fails, bad phi,MAP solution, prior scaling, lam mults 
9) synth restart and upgrade 1.1M par problem"""

ies_vars = ["ies_par_en", "ies_obs_en", "ies_restart_obs_en",
            "ies_bad_phi", "parcov_filename", "ies_num_reals",
            "ies_use_approx", "ies_use_prior_scaling", "ies_reg_factor",
            "ies_lambda_mults", "ies_initial_lambda","ies_include_base","ies_subset_size"]


# the old path system before moving to separate benchmarks repo
# intel = False
# if "windows" in platform.platform().lower():
#     if intel:
#         exe_path = os.path.join("..", "..", "..", "bin", "iwin", "ipestpp-ies.exe")
#     else:
#         exe_path = os.path.join("..", "..", "..", "bin", "win", "pestpp-ies.exe")
# elif "darwin" in platform.platform().lower():
#     exe_path = os.path.join("..", "..", "..", "bin", "mac", "pestpp-ies")
# else:
#     exe_path = os.path.join("..", "..", "..", "bin", "linux", "pestpp-ies")

bin_path = os.path.join("test_bin")
if "linux" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"linux")
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join(bin_path,"mac")
else:
    bin_path = os.path.join(bin_path,"win")

bin_path = os.path.abspath("test_bin")
os.environ["PATH"] += os.pathsep + bin_path

# case of either appveyor, travis or local
if os.path.exists(os.path.join("pestpp","bin")):
    bin_path = os.path.join("..","..","pestpp","bin")
else:
    bin_path = os.path.join("..","..","..","..","pestpp","bin")

        
if "windows" in platform.platform().lower():
    exe_path = os.path.join(bin_path, "iwin", "ipestpp-ies.exe")
elif "darwin" in platform.platform().lower():
    exe_path = os.path.join(bin_path,  "mac", "pestpp-ies")
else:
    exe_path = os.path.join(bin_path, "linux", "pestpp-ies")





noptmax = 3

compare_files = ["pest.phi.actual.csv", "pest.phi.meas.csv", "pest.phi.regul.csv",
                 "pest.{0}.par.csv".format(noptmax), "pest.{0}.obs.csv".format(noptmax),
                 "pest.{0}.par.csv".format(0), "pest.base.obs.csv"]
diff_tol = 1.0e-6
port = 4016
num_reals = 10
def write_empty_test_matrix():
    test_names = [t.split()[0] for t in tests.split('\n')]
    df = pd.DataFrame(index=test_names, columns=ies_vars)
    df.loc[:, "text"] = tests.split('\n')
    df.to_csv("ies_test.blank.csv")


def setup_suite_dir(model_d):
    pyemu.Ensemble.reseed()
    base_d = os.path.join(model_d, "template")
    new_d = os.path.join(model_d, "test_template")
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(base_d, new_d)
    print(platform.platform().lower())
    local=True
    if "linux" in platform.platform().lower() and "10par" in model_d:
        #print("travis_prep")
        #prep_for_travis(model_d)
        local=False
    pst = pyemu.Pst(os.path.join(new_d, "pest.pst"))
    print(pst.model_command)
    
    # set first par as fixed
    #pst.parameter_data.loc[pst.par_names[0], "partrans"] = "fixed"

    pst.observation_data.loc[pst.nnz_obs_names,"weight"] = 1.0

    # set noptmax
    pst.control_data.noptmax = noptmax

    # wipe all pestpp options
    pst.pestpp_options = {}

    # write a generic 2D cov
    if os.path.exists(os.path.join(new_d,"prior.jcb")):
        cov = pyemu.Cov.from_binary(os.path.join(new_d,"prior.jcb"))
        #cov.to_ascii(os.path.join(new_d,"prior.cov"))
    elif os.path.exists(os.path.join(new_d, "prior.cov")):
        cov = pyemu.Cov.from_ascii(os.path.join(new_d, "prior.cov"))
    else:
        cov = pyemu.Cov.from_parameter_data(pst)
        cov = pyemu.Cov(cov.as_2d, names=cov.row_names)
        #cov.to_ascii(os.path.join(new_d, "prior.cov"))
        cov.to_binary(os.path.join(new_d, "prior.jcb"))

    # draw some ensembles
    idx = [i for i in range(num_reals)]
    idx[-1] = "base"
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst, cov=cov, num_reals=num_reals,
                                                    use_homegrown=True,group_chunks=True)
    pe.index = idx
    pe.to_csv(os.path.join(new_d, "par.csv"))
    pe.to_binary(os.path.join(new_d, "par.jcb"))
    pe.to_csv(os.path.join(new_d, "sweep_in.csv"))
    pe.loc[:, pst.adj_par_names].to_csv(os.path.join(new_d, "par_some.csv"))
    pe.iloc[:-3, :].to_csv(os.path.join(new_d, "restart_failed_par.csv"))
    oe = pyemu.ObservationEnsemble.from_id_gaussian_draw(pst, num_reals=num_reals)
    oe.index = idx
    oe.to_csv(os.path.join(new_d, "obs.csv"))
    oe.iloc[:-3, :].to_csv(os.path.join(new_d, "restart_failed_base_obs.csv"))
    oe.to_binary(os.path.join(new_d, "obs.jcb"))

    pst.write(os.path.join(new_d, "pest.pst"))
    # run sweep
    m_d = os.path.join(model_d,"master_sweep1")
    if os.path.exists(m_d):
        shutil.rmtree(m_d)
    pyemu.os_utils.start_workers(new_d, exe_path.replace("-ies","-swp"), "pest.pst", 5, master_dir=m_d,
                           worker_root=model_d,local=local,port=port)
    #shutil.copytree(new_d,m_d)
    #pyemu.os_utils.run("{0} pest.pst".format(exe_path.replace("-ies","-swp")),cwd=m_d)

    # process sweep output as restart csv and jcb
    df = pd.read_csv(os.path.join(m_d, "sweep_out.csv"),index_col=0)
    df.index = df.input_run_id
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(os.path.join(new_d, "restart.csv"))
    df = df.loc[:, pst.nnz_obs_names]
    df.loc[:, pst.nnz_obs_names].to_csv(os.path.join(new_d, "restart_some.csv"))
    df.iloc[:-3, :].to_csv(os.path.join(new_d, "restart_failed.csv"))

    # run pyemu ies
    # pyemu_d = os.path.join(model_d, "master_pyemu")
    # if os.path.exists(pyemu_d):
    #     shutil.rmtree(pyemu_d)
    # shutil.copytree(new_d, pyemu_d)
    # bdir = os.getcwd()
    # os.chdir(pyemu_d)
    # ies = pyemu.EnsembleSmoother("pest.pst", num_slaves=10, verbose="ies.log", slave_dir=os.path.join("..", "template"))
    # ies.initialize(parensemble="par.csv", obsensemble="obs.csv")
    # for i in range(pst.control_data.noptmax):
    #     ies.update()
    # os.chdir(bdir)

    # pestpp_d = os.path.join(model_d,"master")
    # if os.path.exists(pestpp_d):
    #     shutil.rmtree(pestpp_d)
    # shutil.copytree(new_d,pestpp_d)



def freyberg_localizer_test1():

    model_d = "ies_freyberg"
    test_d = os.path.join(model_d, "master_local1")
    template_d = os.path.join(model_d, "template")

    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    print("loading pst")
    pst = pyemu.Pst(os.path.join(template_d, "pest.pst"))
    par = pst.parameter_data
    mat = pyemu.Matrix.from_names(pst.nnz_obs_names,pst.adj_par_names).to_dataframe()
    mat.loc[:,:] = 1.0
    future_pars = par.loc[par.pargp.apply(lambda x: x in ["w1","r1"]),"parnme"]
    mat.loc[:,future_pars] = 0.0
    mat = pyemu.Matrix.from_dataframe(mat)
    mat.to_ascii(os.path.join(template_d,"localizer.mat"))

    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst,cov=cov,num_reals=10)
    pe.enforce()
    pe.to_csv(os.path.join(template_d,"par_local.csv"))
   
    
    pst.observation_data.loc[pst.nnz_obs_names[3],"obgnme"] = "less_than"
    pst.observation_data.loc[pst.nnz_obs_names[3],"weight"] = 100.0
    pst.pestpp_options = {}
    pst.pestpp_options["ies_num_reals"] = 10
    pst.pestpp_options["ies_subset_size"] = 3
    pst.pestpp_options["ies_lambda_mults"] = 1.0
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["ies_include_base"] = False
    pst.pestpp_options["ies_par_en"] = "par_local.csv"
    pst.pestpp_options["ies_localizer"] = "localizer.mat"
    pst.pestpp_options["ies_verbose_level"] = 1
    pst.control_data.noptmax = 3
    print("writing pst")
    pst.write(os.path.join(template_d, "pest_base.pst"))
    print("starting slaves")
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_base.pst", num_slaves=11, master_dir=test_d,
                               worker_root=model_d,port=port)
    par_df = pd.read_csv(os.path.join(test_d,"pest_base.{0}.par.csv".format(pst.control_data.noptmax)),index_col=0)
    #par_df.index = pe.index
    par_df.columns = par_df.columns.str.lower()

    par_df_org = pd.read_csv(os.path.join(test_d, "pest_base.0.par.csv"), index_col=0)
    #par_df_org.index = pe.index
    par_df_org.columns = par_df_org.columns.str.lower()

    broke = []
    for pg in ["r1","w1"]:
        o = par_df_org.loc[:,par.loc[par.pargp==pg,"parnme"]]

        omn,ostd = o.mean(),o.std()
        u = par_df.loc[:,par.loc[par.pargp==pg,"parnme"]]
        for col in o.columns:
            diff = o.loc[:,col] - u.loc[:,col]
            ad = np.abs(diff).sum()
            if ad > 1.0e-5:
                print(col,ad)
                broke.append(col)
        umn,ustd = u.mean(),u.std()
    if len(broke) > 0:
        raise Exception("future pars too diff:{0}".format(','.join(broke)))


def freyberg_localizer_test2():
    model_d = "ies_freyberg"
    test_d = os.path.join(model_d, "master_local2")
    template_d = os.path.join(model_d, "template")

    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    print("loading pst")
    pst = pyemu.Pst(os.path.join(template_d, "pest.pst"))
    par = pst.parameter_data
    mat = pyemu.Matrix.from_names(pst.nnz_obs_names, pst.adj_par_names).to_dataframe()
    mat.loc[:, :] = 1.0
    future_pars = par.loc[par.pargp.apply(lambda x: x in ["w1", "r1"]), "parnme"]
    mat.loc[:, future_pars] = 0.0
    mat = pyemu.Matrix.from_dataframe(mat)
    mat.to_ascii(os.path.join(template_d, "localizer.mat"))

    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst, cov=cov, num_reals=10)
    pe.enforce()
    pe.to_csv(os.path.join(template_d, "par_local.csv"))

    pst.observation_data.loc[pst.nnz_obs_names[3], "obgnme"] = "less_than"
    pst.observation_data.loc[pst.nnz_obs_names[3], "weight"] = 100.0
    pst.pestpp_options = {}
    pst.pestpp_options["ies_num_reals"] = 10
    pst.pestpp_options["ies_subset_size"] = 3
    #pst.pestpp_options["ies_lambda_mults"] = 1.0
    #pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["ies_include_base"] = False
    pst.pestpp_options["ies_par_en"] = "par_local.csv"
    pst.pestpp_options["ies_localizer"] = "localizer.mat"
    pst.pestpp_options["ies_verbose_level"] = 1
    pst.control_data.nphistp = 10
    pst.control_data.nphinored = 10
    pst.control_data.noptmax = 3
    print("writing pst")
    pst.write(os.path.join(template_d, "pest_local.pst"))
    print("starting slaves")
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_local.pst", num_slaves=11, master_dir=test_d,
                               worker_root=model_d, port=port)
    par_df1 = pd.read_csv(os.path.join(test_d, "pest_local.{0}.par.csv".format(pst.control_data.noptmax)), index_col=0)
    #par_df1.index = pe.index
    par_df1.columns = par_df1.columns.str.lower()
    phi_df1 = pd.read_csv(os.path.join(test_d, "pest_local.phi.composite.csv"))

    pst.pestpp_options.pop("ies_localizer")
    pst.write(os.path.join(template_d, "pest_base.pst"))
    print("starting slaves")
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_base.pst", num_slaves=11, master_dir=test_d+"_base",
                               worker_root=model_d, port=port)
    par_df2 = pd.read_csv(os.path.join(test_d+"_base", "pest_base.{0}.par.csv".format(pst.control_data.noptmax)), index_col=0)
    #par_df2.index = pe.index
    par_df2.columns = par_df2.columns.str.lower()
    phi_df2 = pd.read_csv(os.path.join(test_d, "pest_local.phi.composite.csv"))
    plt.plot(phi_df1.total_runs, phi_df1.loc[:, "mean"], label="local")
    plt.plot(phi_df2.total_runs, phi_df2.loc[:, "mean"], label="full")
    plt.legend()
    plt.savefig(os.path.join(test_d, "local_test.pdf"))


def freyberg_localizer_test3():
    """freyberg local 3"""
    model_d = "ies_freyberg"
    test_d = os.path.join(model_d, "master_local31")
    template_d = os.path.join(model_d, "template")

    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    print("loading pst")
    pst = pyemu.Pst(os.path.join(template_d, "pest.pst"))
    par = pst.parameter_data
    mat = pyemu.Matrix.from_names(pst.nnz_obs_names, pst.par_groups).to_dataframe()
    mat.loc[:, :] = 1.0
    #future_pars = par.loc[par.pargp.apply(lambda x: x in ["w1", "r1"]), "parnme"]
    mat.loc[:, ["w1","r1"]] = 0.0
    mat = pyemu.Matrix.from_dataframe(mat)
    mat.to_ascii(os.path.join(template_d, "localizer.mat"))

    cov = pyemu.Cov.from_parameter_data(pst)
    pe = pyemu.ParameterEnsemble.from_gaussian_draw(pst=pst, cov=cov, num_reals=10)
    pe.enforce()
    pe.to_csv(os.path.join(template_d, "par_local.csv"))

    pst.observation_data.loc[pst.nnz_obs_names[3], "obgnme"] = "less_than"
    pst.observation_data.loc[pst.nnz_obs_names[3], "weight"] = 100.0
    pst.pestpp_options = {}
    pst.pestpp_options["ies_num_reals"] = 10
    pst.pestpp_options["ies_subset_size"] = 3
    pst.pestpp_options["ies_lambda_mults"] = 1.0
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["ies_include_base"] = False
    pst.pestpp_options["ies_par_en"] = "par_local.csv"
    pst.pestpp_options["ies_localizer"] = "localizer.mat"
    pst.pestpp_options["ies_verbose_level"] = 1
    pst.pestpp_options["ies_localize_how"] = "par"
    pst.control_data.noptmax = 3
    print("writing pst")
    pst.write(os.path.join(template_d, "pest_local.pst"))
    print("starting slaves")
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_local.pst", num_slaves=15, master_dir=test_d,
                               worker_root=model_d, port=port)
    par_df1 = pd.read_csv(os.path.join(test_d, "pest_local.{0}.par.csv".format(pst.control_data.noptmax)), index_col=0)
    #par_df1.index = pe.index
    par_df1.columns = par_df1.columns.str.lower()
    phi_df1 = pd.read_csv(os.path.join(test_d, "pest_local.phi.composite.csv"))

    pst.pestpp_options.pop("ies_localizer")
    pst.write(os.path.join(template_d, "pest_base.pst"))
    print("starting slaves")
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_base.pst", num_slaves=15, master_dir=test_d+"_base",
                               worker_root=model_d, port=port)
    par_df2 = pd.read_csv(os.path.join(test_d+"_base", "pest_base.{0}.par.csv".format(pst.control_data.noptmax)), index_col=0)
    #par_df2.index = pe.index
    par_df2.columns = par_df2.columns.str.lower()
    phi_df2 = pd.read_csv(os.path.join(test_d, "pest_local.phi.composite.csv"))
    plt.plot(phi_df1.total_runs, phi_df1.loc[:, "mean"], label="local")
    plt.plot(phi_df2.total_runs, phi_df2.loc[:, "mean"], label="full")
    plt.legend()
    plt.savefig(os.path.join(test_d, "local_test.pdf"))
    diff = np.abs(phi_df1.loc[:, "mean"] - phi_df2.loc[:, "mean"])
    assert diff.max() < 1.0e-5,diff

def compare_freyberg_local3():
    df_l = pd.read_csv(os.path.join("ies_freyberg","master_local3","pest_local.6.par.csv"))
    df_f = pd.read_csv(os.path.join("ies_freyberg","master_local3_base","pest_base.6.par.csv"))

    pst = pyemu.Pst(os.path.join("ies_freyberg","template","pest.pst"))
    df_l.columns = df_l.columns.str.lower()
    df_f.columns = df_f.columns.str.lower()
    groups = ["w1","r1"]
    par = pst.parameter_data
    # pars = par.loc[par.pargp.apply(lambda x: x in groups)]
    # df_l = df_l.loc[:,pars]
    # df_f = df_f.loc[:,pars]
    for group,title in zip(groups,["future pumping","future recharge"]):
        fig = plt.figure(figsize=(10,10))
        ax1,ax2 = plt.subplot(211), plt.subplot(212)
        fig.suptitle(title+ " parameters with (blue) and w/o (green) localization")
        ax1.set_title("mean")
        ax2.set_title("standard deviation")

        for ax in zip(groups,[ax1,ax2]):
            pars = par.loc[par.pargp==group,"parnme"]
            df_lg = df_l.loc[:, pars]
            df_fg = df_f.loc[:, pars]
            mn_l, st_l = df_lg.mean(),df_lg.std()
            mn_f, st_f = df_fg.mean(), df_fg.std()

            print(mn_l,mn_f)
            print(st_l,st_f)
            mn_l.hist(ax=ax1,facecolor='b',alpha=0.5,normed=True)
            mn_f.hist(ax=ax1,facecolor='g',alpha=0.5,normed=True)
            st_l.hist(ax=ax2, facecolor='b', alpha=0.5,normed=True)
            st_f.hist(ax=ax2, facecolor='g', alpha=0.5,normed=True)
        ax2.set_yticklabels([])
        ax1.set_yticklabels([])

        plt.show()


def csv_tests():
    """csv tests"""
    model_d = "ies_10par_xsec"
    test_d = os.path.join(model_d, "master_csv_test1")
    template_d = os.path.join(model_d, "test_template")
    if not os.path.exists(template_d):
        raise Exception("template_d {0} not found".format(template_d))
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    # shutil.copytree(template_d, test_d)
    pst_name = os.path.join(template_d, "pest.pst")
    pst = pyemu.Pst(pst_name)

    # mat = pyemu.Matrix.from_names(pst.nnz_obs_names,pst.adj_par_names).to_dataframe()
    mat = pyemu.Matrix.from_names(["head"], pst.adj_par_names).to_dataframe()
    mat.loc[:, :] = 1.0
    mat.to_csv(os.path.join(template_d,"localizer.csv"))
    mat = pyemu.Matrix.from_dataframe(mat)
    mat.to_ascii(os.path.join(template_d, "localizer.mat"))
    cov = pyemu.Cov.from_parameter_data(pst)
    cov.to_dataframe().to_csv(os.path.join(template_d,"prior.csv"))
    pst.pestpp_options = {}
    pst.pestpp_options["ies_num_reals"] = 10
    pst.pestpp_options["ies_localizer"] = "localizer.csv"
    pst.pestpp_options["ies_lambda_mults"] = 1.0
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["ies_subset_size"] = 11
    pst.pestpp_options["parcov_filename"] = "prior.csv"
    pst.pestpp_options["ies_use_prior_scaling"] = True
    pst.pestpp_options["ies_use_approx"] = False
    pst.control_data.noptmax = 3
    pst_name = os.path.join(template_d, "pest_local.pst")
    pst.write(pst_name)
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_local.pst", num_slaves=11,
                               master_dir=test_d, verbose=True, worker_root=model_d,
                               port=4019)
    # phi_df1 = pd.read_csv(os.path.join(test_d, "pest_local.phi.actual.csv"))
    #
    # pst.pestpp_options.pop("ies_localizer")
    # pst.write(pst_name)
    # pyemu.os_utils.start_workers(template_d, exe_path, "pest_local.pst", num_slaves=11,
    #                            master_dir=test_d, verbose=True, worker_root=model_d,
    #                            port=port)
    # phi_df2 = pd.read_csv(os.path.join(test_d, "pest_local.phi.actual.csv"))
    # diff = phi_df1 - phi_df2
    # print(diff.max().max())
    # assert diff.max().max() == 0
    # plt.plot(phi_df1.total_runs, phi_df1.loc[:, "mean"], label="local")
    # plt.plot(phi_df2.total_runs, phi_df2.loc[:, "mean"], label="full")
    # plt.legend()
    # plt.savefig(os.path.join(test_d, "local_test.pdf"))

#def tenpar_include_base_test():

def tenpar_restart_binary_test():
    """tenpar restart tests"""
    model_d = "ies_10par_xsec"
    test_d = os.path.join(model_d, "master_binary_restart1")
    template_d = os.path.join(model_d, "template")
    pst = pyemu.Pst(os.path.join(template_d, "pest.pst"))
    num_reals = 10
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    # shutil.copytree(template_d, test_d)
    pst.pestpp_options = {"ies_num_reals": num_reals}
    pst.pestpp_options["ies_save_binary"] = True
    pst.pestpp_options["ies_lambda_mults"] = 1.0
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["ies_include_base"] = False
    pst.control_data.noptmax = 1
    pst.write(os.path.join(template_d, "pest_restart.pst"))
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_restart.pst", num_slaves=10,
                                worker_root=model_d, master_dir=test_d, port=port)
    # pyemu.os_utils.run("{0} {1}".format(exe_path, "pest_restart.pst"), cwd=test_d)
    for f in ["pest_restart.0.par.jcb","pest_restart.base.obs.jcb","pest_restart.0.obs.jcb"]:
        shutil.copy2(os.path.join(test_d,f),os.path.join(template_d,f))
    oe = pyemu.ObservationEnsemble.from_binary(pst,os.path.join(test_d,"pest_restart.0.obs.jcb"))
    pe = pyemu.ParameterEnsemble.from_binary(pst, os.path.join(test_d, "pest_restart.0.par.jcb"))
    df1 = pd.read_csv(os.path.join(test_d,"pest_restart.phi.actual.csv"),index_col=0)
    assert oe.shape == (num_reals,pst.nobs)
    assert pe.shape == (num_reals,pst.npar)

    assert os.path.exists(os.path.join(test_d, "pest_restart.phi.group.csv"))
    df = pd.read_csv(os.path.join(test_d, "pest_restart.phi.group.csv"))
    diff = df.obs_realization.apply(np.int) - df.par_realization.apply(np.int)
    assert diff.max() == 0, diff

    pst.pestpp_options["ies_par_en"] = "pest_restart.0.par.jcb"
    pst.pestpp_options["ies_obs_en"] = "pest_restart.base.obs.jcb"
    pst.pestpp_options["ies_restart_obs_en"] = "pest_restart.0.obs.jcb"
    pst.write(os.path.join(template_d, "pest_restart_2.pst"))
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_restart_2.pst", num_slaves=10,
                                worker_root=model_d, master_dir=test_d+"_2", port=port)
    pe1 = pyemu.ParameterEnsemble.from_binary(pst, os.path.join(test_d+"_2", "pest_restart_2.0.par.jcb"))
    oe1 = pyemu.ObservationEnsemble.from_binary(pst, os.path.join(test_d+"_2", "pest_restart.0.obs.jcb"))

    pe_diff = pe1 - pe
    oe_diff = oe1 - oe
    print(pe1)
    print(pe)
    #assert pe_diff.max().max() == 0.0, pe_diff.max()
    #assert oe_diff.max().max() == 0.0, oe_diff.max()
    df2 = pd.read_csv(os.path.join(test_d+"_2", "pest_restart_2.phi.actual.csv"), index_col=0)
    assert oe1.shape == (num_reals, pst.nobs)
    assert pe1.shape == (num_reals, pst.npar)
    diff = df1.loc[0,"mean"] - df2.loc[0,"mean"]
    assert diff == 0.0,diff
    assert os.path.exists(os.path.join(test_d+"_2", "pest_restart_2.phi.group.csv"))
    df = pd.read_csv(os.path.join(test_d+"_2", "pest_restart_2.phi.group.csv"))
    diff = df.obs_realization - df.par_realization
    assert diff.max() == 0, diff



    #
    # shutil.copy2(os.path.join(test_d, "pest_restart.base.obs.csv"), os.path.join(template_d, "base.csv"))
    #
    # pst.pestpp_options = {}
    # pst.pestpp_options["ies_par_en"] = "par1.csv"
    # pst.pestpp_options["ies_lambda_mults"] = 1.0
    # pst.pestpp_options["lambda_scale_fac"] = 1.0
    # # pst.pestpp_options["ies_num_reals"] = num_reals
    # pst.pestpp_options["ies_restart_obs_en"] = "restart1.csv"
    # pst.pestpp_options["ies_obs_en"] = "base.csv"
    # pst.control_data.noptmax = 3
    # pst.write(os.path.join(template_d, "pest_restart.pst"))
    # pyemu.os_utils.start_workers(template_d, exe_path, "pest_restart.pst", num_slaves=10,
    #                             worker_root=model_d, master_dir=test_d, port=port)
    # assert os.path.exists(os.path.join(test_d, "pest_restart.3.par.csv"))

def tenpar_restart_test():
    """tenpar restart tests"""
    model_d = "ies_10par_xsec"
    test_d = os.path.join(model_d, "master_restart1")
    template_d = os.path.join(model_d, "template")
    pst = pyemu.Pst(os.path.join(template_d, "pest.pst"))
    num_reals = 30
    if os.path.exists(test_d):
       shutil.rmtree(test_d)
    #shutil.copytree(template_d, test_d)
    pst.pestpp_options = {"ies_num_reals":num_reals}
    pst.control_data.noptmax = -1
    pst.write(os.path.join(template_d,"pest_restart.pst"))
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_restart.pst", num_slaves=10,
                                worker_root=model_d, master_dir=test_d, port=port)
    #pyemu.os_utils.run("{0} {1}".format(exe_path, "pest_restart.pst"), cwd=test_d)

    par_df = pd.read_csv(os.path.join(test_d,"pest_restart.0.par.csv"),index_col=0)
    par_df = par_df.iloc[::2,:]
    par_df.to_csv(os.path.join(template_d,"par1.csv"))

    obs_df = pd.read_csv(os.path.join(test_d,"pest_restart.0.obs.csv"),index_col=0)
    obs_df = obs_df.iloc[::2,:]
    obs_df.to_csv(os.path.join(template_d,"restart1.csv"))

    shutil.copy2(os.path.join(test_d,"pest_restart.base.obs.csv"),os.path.join(template_d,"base.csv"))

    pst.pestpp_options = {}
    pst.pestpp_options["ies_par_en"] = "par1.csv"
    pst.pestpp_options["ies_lambda_mults"] = 1.0
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["ies_debug_fail_subset"] = True
    pst.pestpp_options["ies_debug_fail_remainder"] = True
    pst.pestpp_options["ies_debug_bad_phi"] = True
    #pst.pestpp_options["ies_num_reals"] = num_reals
    pst.pestpp_options["ies_restart_obs_en"] = "restart1.csv"
    pst.pestpp_options["ies_obs_en"] = "base.csv"
    pst.control_data.noptmax = 2
    pst.write(os.path.join(template_d,"pest_restart.pst"))
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_restart.pst", num_slaves=10,
                                worker_root=model_d, master_dir=test_d, port=port)
    assert os.path.exists(os.path.join(test_d,"pest_restart.{0}.par.csv".format(pst.control_data.noptmax))),\
        os.listdir(test_d)
    assert os.path.exists(os.path.join(test_d, "pest_restart.phi.group.csv"))
    df = pd.read_csv(os.path.join(test_d, "pest_restart.phi.group.csv"))
    diff = df.obs_realization - df.par_realization
    assert diff.max() == 0,diff

def tenpar_par_restart_test():
    """tenpar par restart tests"""
    model_d = "ies_10par_xsec"

    template_d = os.path.join(model_d, "template")
    pst = pyemu.Pst(os.path.join(template_d, "pest.pst"))
    num_reals = 30
    test_d = os.path.join(model_d, "master_easy_restart")
    if os.path.exists(test_d):
       shutil.rmtree(test_d)
    #shutil.copytree(template_d, test_d)
    pst.pestpp_options = {}
    pst.pestpp_options = {"ies_num_reals":num_reals}
    pst.pestpp_options["ies_include_base"] = False
    pst.pestpp_options["ies_subset_how"] = "first"
    pst.pestpp_options["ies_accept_phi_fac"] = 1.0
    pst.pestpp_options["ies_lambda_mults"] = 1.0
    pst.pestpp_options["lambda_scale_fac"] = 1.0
    pst.pestpp_options["ies_lambda_dec_fac"] = 1.0
    pst.pestpp_options["ies_init_lam"] = 10.0
    pst.pestpp_options["ies_save_lambda_en"] = True
    pst.control_data.noptmax = 3
    pst.write(os.path.join(template_d,"pest_restart.pst"))
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_restart.pst", num_slaves=10,
                                worker_root=model_d, master_dir=test_d, port=port)
    #pyemu.os_utils.run("{0} {1}".format(exe_path, "pest_restart.pst"), cwd=test_d)

    par_df = pd.read_csv(os.path.join(test_d,"pest_restart.1.par.csv"),index_col=0)
    #par_df = par_df.iloc[::2,:]
    par_df.to_csv(os.path.join(template_d,"par1.csv"))

    obs_df = pd.read_csv(os.path.join(test_d,"pest_restart.1.obs.csv"),index_col=0)
    #obs_df = obs_df.iloc[::2,:]
    obs_df.to_csv(os.path.join(template_d,"restart1.csv"))

    shutil.copy2(os.path.join(test_d,"pest_restart.base.obs.csv"),os.path.join(template_d,"base.csv"))
    shutil.copy2(os.path.join(test_d, "pest_restart.0.par.csv"), os.path.join(template_d, "par_base.csv"))

    #pst.pestpp_options = {}
    pst.pestpp_options["ies_par_en"] = "par_base.csv"
    pst.pestpp_options["ies_restart_par_en"] = "par1.csv"
    # pst.pestpp_options["ies_lambda_mults"] = 1.0
    # pst.pestpp_options["lambda_scale_fac"] = 1.0
    # pst.pestpp_options["ies_lambda_dec_fac"] = 1.0
    # #pst.pestpp_options["ies_debug_fail_subset"] = True
    #pst.pestpp_options["ies_debug_fail_remainder"] = True
    #pst.pestpp_options["ies_debug_bad_phi"] = True
    #pst.pestpp_options["ies_num_reals"] = num_reals
    pst.pestpp_options["ies_restart_obs_en"] = "restart1.csv"
    pst.pestpp_options["ies_obs_en"] = "base.csv"
    pst.control_data.noptmax = 2
    pst.write(os.path.join(template_d,"pest_restart.pst"))
    test_d = os.path.join(model_d, "master_easy_restart_withpar")
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_restart.pst", num_slaves=10,
                                worker_root=model_d, master_dir=test_d, port=port)
    assert os.path.exists(os.path.join(test_d,"pest_restart.{0}.par.csv".format(pst.control_data.noptmax))),\
        os.listdir(test_d)
    assert os.path.exists(os.path.join(test_d, "pest_restart.phi.group.csv"))
    df = pd.read_csv(os.path.join(test_d, "pest_restart.phi.group.csv"))
    diff = df.obs_realization - df.par_realization
    assert diff.max() == 0,diff

    phi_df1 = pd.read_csv(os.path.join(test_d, "pest_restart.phi.actual.csv"),index_col=0)
    pst.pestpp_options.pop("ies_restart_par_en")
    pst.pestpp_options["ies_par_en"] = "par1.csv"
    pst.write(os.path.join(template_d, "pest_restart.pst"))
    test_d = os.path.join(model_d, "master_easy_restart_nopar")
    if os.path.exists(test_d):
        shutil.rmtree(test_d)
    pyemu.os_utils.start_workers(template_d, exe_path, "pest_restart.pst", num_slaves=10,
                                worker_root=model_d, master_dir=test_d, port=port)
    phi_df2 = pd.read_csv(os.path.join(test_d, "pest_restart.phi.actual.csv"),index_col=0)
    diff = (phi_df1 - phi_df2).apply(np.abs)
    print(diff.max())
    assert diff.max().max()==0.0,diff.max().max()




def tenpar_rns_test():
    """tenpar rns test"""
    model_d = "ies_10par_xsec"
    test_d = os.path.join(model_d, "master_rns")
    template_d = os.path.join(model_d, "template")
    pst = pyemu.Pst(os.path.join(template_d, "pest.pst"))

    num_reals = 5
    if os.path.exists(test_d):
       shutil.rmtree(test_d)
    shutil.copytree(template_d, test_d)
    pst.pestpp_options = {"ies_num_reals":num_reals}
    pst.control_data.noptmax = -1
    pst.write(os.path.join(test_d,"pest_restart.pst"))
    pyemu.os_utils.run("{0} {1}".format(exe_path, "pest_restart.pst"), cwd=test_d)

    obs_df1 = pd.read_csv(os.path.join(test_d,"pest_restart.0.obs.csv"),index_col=0)
    assert obs_df1.shape[0] == num_reals,obs_df1.shape
    pst.pestpp_options = {}
    num_reals = 10
    pst.pestpp_options["ies_num_reals"] = num_reals
    pst.write(os.path.join(test_d,"pest_restart.pst"))
    pyemu.os_utils.run("{0} {1}".format(exe_path, "pest_restart.pst"), cwd=test_d)
    obs_df2 = pd.read_csv(os.path.join(test_d,"pest_restart.0.obs.csv"),index_col=0)
    assert obs_df2.shape[0] == num_reals,obs_df2.shape
    




if __name__ == "__main__":
    #write_empty_test_matrix()

    # setup_suite_dir("ies_10par_xsec")
    # setup_suite_dir("ies_freyberg")
    # run_suite("ies_10par_xsec")
    # run_suite("ies_freyberg")
    # rebase("ies_freyberg")
    # rebase("ies_10par_xsec")
    # compare_suite("ies_10par_xsec")
    # compare_suite("ies_freyberg")
    #eval_freyberg()
    #eval_10par_xsec()

    #full list of tests
    # tenpar_subset_test()
    # tenpar_full_cov_test()
    # eval_freyberg_full_cov_reorder()
    # test_freyberg_full_cov_reorder_run()
    # eval_freyberg_full_cov()
    # tenpar_tight_tol_test()
    # test_chenoliver()
    # tenpar_narrow_range_test()
    # test_freyberg_ineq()
    # tenpar_fixed_test()
    # tenpar_fixed_test2()\
    # tenpar_subset_how_test()
    # tenpar_localizer_test1()
    #tenpar_localizer_test2()
    # tenpar_localizer_test3()
    freyberg_localizer_test1()
    # freyberg_localizer_eval2()
    # freyberg_localizer_test3()
    # freyberg_dist_local_test()
    # freyberg_local_threads_test()
    # tenpar_restart_binary_test()
    # tenpar_restart_test()
    # csv_tests()
    # tenpar_rns_test()
    # clues_longnames_test()
    # tenpar_localize_how_test()
    #tenpar_incr_num_reals_test()
    #freyberg_dist_local_invest()
    #tenpar_tied_test()
    #tenpar_by_vars_test()
    #tenpar_par_restart_test()