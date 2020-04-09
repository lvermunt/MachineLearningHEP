#!/bin/bash

#SBATCH --output=slurm-%J.out
#SBATCH --error=slurm-%J.out

function die
{
    echo $1
    exit
}

JOBIDX="-0"
[ -n "${SLURM_ARRAY_TASK_ID}" ] && JOBIDX="-${SLURM_ARRAY_TASK_ID}"
export JOBIDX

unset DISPLAY
export MLPBACKEND=pdf
#mv -v /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodDs/261_20191004-0007/pk* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodDs/261_20191004-0007/
#mv -v /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodDs/261_20191004-0007/pk* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodDs/261_20191004-0007/
#mv -v /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodDs/261_20191004-0007/pk* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodDs/261_20191004-0007/

cp -vr /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodDs/261_20191004-0007/skpklde* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodDs/261_20191004-0007/
cp -vr /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodDs/261_20191004-0007/skpklde* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodDs/261_20191004-0007/
cp -vr /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodDs/261_20191004-0007/skpklde* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodDs/261_20191004-0007/

#mv -v /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_data_evttot /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/
#mv -v /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_mc_prodDs_evttot /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/

#mv -v /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007/pk* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007/
#mv -v /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008/pk* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008/
#mv -v /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/pk* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/

cp -vr /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007/skpklde* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007/
cp -vr /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008/skpklde* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008/
cp -vr /data/LuukTempMarch/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpklde* /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/

#mv -v /data/LuukTempMarch/D0DsLckINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007/nonunique_removed /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/TTree/D0DsLckINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007/nonunique_removed
#mv -v /data/LuukTempMarch/D0DsLckINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008/nonunique_removed /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/TTree/D0DsLckINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008/nonunique_removed
#mv -v /data/LuukTempMarch/D0DsLckINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/nonunique_removed /mnt/temp/QM19_Ds_vsMult_Preliminaries/EvIDbug_fixed_March2020/TTree/D0DsLckINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/nonunique_removed

#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodDs/261_20191004-0007/skpkldec/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodDs/261_20191004-0007
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodDs/261_20191004-0007/skpkldecmerged/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodDs/261_20191004-0007
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodDs/261_20191004-0007/skpkldec/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodDs/261_20191004-0007
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodDs/261_20191004-0007/skpkldecmerged/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodDs/261_20191004-0007
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodDs/261_20191004-0007/skpkldec/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodDs/261_20191004-0007
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodDs/261_20191004-0007/skpkldecmerged/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodDs/261_20191004-0007
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007/skpkldec/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007/skpkldecmerged/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_data/258_20191004-0007
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008/skpkldec/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008/skpkldecmerged/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_data/259_20191004-0008
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldec/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008
#cp -avr /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/ /data/LuukTempMarch/DerivedQM/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008


#mv /data/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1_opt/pp_2016_mc_prodD2H/261_20191004-0007/* /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodD2H/261_20191004-0007/
#mv /data/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1_opt/pp_2017_mc_prodD2H/261_20191004-0007/* /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodD2H/261_20191004-0007/
#mv /data/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1_opt/pp_2018_mc_prodD2H/261_20191004-0007/* /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/
#mv /data/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodD2H/261_20191004-0007/* /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2016_mc_prodD2H/261_20191004-0007/
#mv /data/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodD2H/261_20191004-0007/* /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2017_mc_prodD2H/261_20191004-0007/
#mv /data/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/* /mnt/temp/QM19_Ds_vsMult_Preliminaries/Derived/DskINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/
#mv /data/DerivedResults/LcPbPb/vAN-20191204_ROOT6-1 /mnt/temp/OngoingAnalysis_LcPbPb/DerivedResults_LcPbPb_vAN-20191204_ROOT6-1
#mv /data/DerivedResults/LcPbPb/vAN-20191209_ROOT6-1 /mnt/temp/OngoingAnalysis_LcPbPb/DerivedResults_LcPbPb_vAN-20191209_ROOT6-1
#mv /data/DerivedResults/LcPbPb/vAN-20191227_ROOT6-1 /mnt/temp/OngoingAnalysis_LcPbPb/DerivedResults_LcPbPb_vAN-20191227_ROOT6-1
#mv /mnt/temp/MovedForTemporarySpace/Derived_LcPbPb_vAN-20191209_ROOT6-1_LHC18pq_319_20191209-2229 /mnt/temp/OngoingAnalysis_LcPbPb/Derived_LcPbPb_vAN-20191209_ROOT6-1_LHC18pq_319_20191209-2229
#mv /data/Derived/LcPbPb/vAN-20191204_ROOT6-1 /mnt/temp/OngoingAnalysis_LcPbPb/Derived_LcPbPb_vAN-20191204_ROOT6-1
#mv /data/Derived/LcPbPb/vAN-20191209_ROOT6-1 /mnt/temp/OngoingAnalysis_LcPbPb/Derived_LcPbPb_vAN-20191209_ROOT6-1
#mv /data/Derived/LcPbPb/vAN-20191227_ROOT6-1 /mnt/temp/OngoingAnalysis_LcPbPb/Derived_LcPbPb_vAN-20191227_ROOT6-1
#mv /data/Derived/BskAnyITS2/vAN-20191228_ROOT6-1 /mnt/temp/MovedForTemporarySpace_28Feb20/Derived_BskAnyITS2_vAN-20191228_ROOT6-1
#mv /data/Derived/BskAnyITS3Improver/vAN-20191228_ROOT6-1 /mnt/temp/MovedForTemporarySpace_28Feb20/Derived_BskAnyITS3Improver_vAN-20191228_ROOT6-1
#mv /data/DerivedResults/BskAnyITS2/vAN-20191228_ROOT6-1 /mnt/temp/MovedForTemporarySpace_28Feb20/DerivedResults_BskAnyITS2_vAN-20191228_ROOT6-1
#mv /data/DerivedResults/BskAnyITS3Improver/vAN-20191228_ROOT6-1 /mnt/temp/MovedForTemporarySpace_28Feb20/DerivedResults_BskAnyITS3Improver_vAN-20191228_ROOT6-1
#mv /home/lvermunt/PIDetafix /mnt/temp/MovedForTemporarySpace_28Feb20/HomeLuuk_PIDetafix
#mv /data/TTree/BskAnyITS2ITS3Improver/vAN-20191228_ROOT6-1/ITS2_19h1b2_full/331_20191229-2130/merged /mnt/temp/MovedForTemporarySpace_20Jan20/TTree_BskAnyITS2ITS3Improver/vAN-20191228_ROOT6-1_ITS2_19h1b2_full/331_20191229-2130/merged
#mv /data/TTree/BskAnyITS2ITS3Improver/vAN-20191228_ROOT6-1/ITS2_19h1b2_full/330_20191229-1514/merged /mnt/temp/MovedForTemporarySpace_20Jan20/TTree_BskAnyITS2ITS3Improver/vAN-20191228_ROOT6-1_ITS2_19h1b2_full/330_20191229-1514/merged
#mv /data/TTree/LcPbPb/vAN-20191204_ROOT6-1/LHC15o/314_20191205-1357/merged /mnt/temp/MovedForTemporarySpace_20Jan20/TTree_LcPbPb/vAN-20191204_ROOT6-1_LHC15o/314_20191205-1357/merged
#mv /data/TTree/LcPbPb/vAN-20191209_ROOT6-1/LHC18pq/319_20191209-2229/merged /mnt/temp/MovedForTemporarySpace_20Jan20/TTree_LcPbPb/vAN-20191209_ROOT6-1_LHC18pq/319_20191209-2229/merged
#mv /data/TTree/LbkAnyITS2ITS3Improver/vAN-20190923_ROOT6-1/ITS2_13d19/247_20190924-0053/merged /mnt/temp/MovedForTemporarySpace_20Jan20/TTree_LbkAnyITS2ITS3Improver/vAN-20190923_ROOT6-1_ITS2_13d19/247_20190924-0053/merged
#mv /data/TTree/LbkAnyITS2ITS3Improver/vAN-20190923_ROOT6-1/ITS2_13d19/248_20190924-0054/merged /mnt/temp/MovedForTemporarySpace_20Jan20/TTree_LbkAnyITS2ITS3Improver/vAN-20190923_ROOT6-1_ITS2_13d19/248_20190924-0054/merged
