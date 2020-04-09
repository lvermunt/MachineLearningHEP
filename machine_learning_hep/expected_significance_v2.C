void fit_signal(TH1F* hsig, int j, int i, double &xmin, double &xmax);
void calculate_background(TH1F* hbkg, TF1* f1, int j, int i, double xmin, double xmax, double &bkgcent);
void extract_fonll(TString filnam, int j, double &fonllmin, double &fonllcent, double &fonllmax);
void extract_TAMU(TString filn, int j, double &tamucent);
void calculate_efficiency(TH1F* hgen, TH1F* hsel, int j, int i, double &effcent, double &erreffcent);

const Int_t nptbins = 7;
Int_t ptbins[nptbins+1] = {0, 2, 4, 6, 8, 12, 16, 24};
Int_t rebin[nptbins] = {15, 15, 15, 15, 16, 18, 20};
Bool_t bincountBkg[nptbins] = {kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE};

Double_t nEv335 = 852644;
TString nEv335String = "8.53 * 10^{5}"; //To confirm for train 335 (05/02/20)
Double_t nEvExpected = 8000000000;
TString nEvExpectedString = "8 * 10^{9}";
TString nLumiExpectedString = "10 nb^{-1}";

TString filenameBkgCorr = "/home/lvermunt/inputBs/BkgCorrFactor_Bs_1DataFile_25MCFile.root";
TString filenameTAMU = "/home/lvermunt/inputBs/input_RAA_TAMU_Bs.txt";
TString filnameFONLL = "/home/lvermunt/inputBs/inputFONLL.txt";
Double_t fbtoB = 0.407; //http://pdg.lbl.gov/2019/reviews/rpp2018-rev-b-meson-prod-decay.pdf, table 85.1
Double_t fbtoBUnc = 0.007;
Double_t fLHCbBBs = 2*0.122; //https://journals.aps.org/prd/pdf/10.1103/PhysRevD.100.031102, factor 2 because B0 + B+
Double_t fLHCbBBsUnc = 2*0.006;

Double_t BRBs = 0.00304;
Double_t errBRBs = 0.00023;
Double_t BRDs = 0.0227;
Double_t errBRDs = 0.0008;
Double_t TAA = 23.07 * 1e-3; //mb^-1 -> mub^-1 in which we put FONLL

Double_t gauss3sigmafactor = 0.9973;

void expected_significance_v2(TString filenamemass, TString filenameeff, TString filenameout, const Int_t trials /*0 = STD*/, Bool_t pid = kFALSE){
  
  TGaxis::SetMaxDigits(3);

  TFile* fmass = new TFile(filenamemass.Data());
  TFile* feff = new TFile(filenameeff.Data());
  TFile* fBkgCorr = new TFile(filenameBkgCorr.Data());

  TString nameeffgen = "h_gen_pr_";
  TString nameeffsel = "h_sel_pr_";
  if(pid){
    nameeffsel = "h_sel_pr_pid_";
  }
  TH1F *hgen[trials+1];
  TH1F *hsel[trials+1];

  TString namemasssig = "hmass_sigpt_cand";
  TString namemassbkg = "hmass_bkgpt_cand";
  TString suffixSTD = "n_tracklets_-999999.00_999999.00";
  if(pid){
    namemasssig = "hmass_pid_sigpt_cand";
    namemassbkg = "hmass_pid_bkgpt_cand";
  }
  
  TH1F *hsig[nptbins][trials+1];
  TH1F *hbkg[nptbins][trials+1];
  TF1  *fbkg[nptbins][trials+1];
  TH1F *hbkgentries[nptbins][trials+1];

  for(int i = 0; i <= trials; i++){
    if(trials != 0){
      hgen[i] = (TH1F*)feff->Get(Form("%s%d",nameeffgen.Data(),i));
      hsel[i] = (TH1F*)feff->Get(Form("%s%d",nameeffsel.Data(),i));

      for(int j = 0; j < nptbins; j++){
        hsig[j][i] = (TH1F*)fmass->Get(Form("%s%d_%d_%d",namemasssig.Data(),ptbins[j],ptbins[j+1],i));
        hbkg[j][i] = (TH1F*)fmass->Get(Form("%s%d_%d_%d",namemassbkg.Data(),ptbins[j],ptbins[j+1],i));
        fbkg[j][i] = new TF1(Form("f_%d_%d",j,i), "expo",  5.07, 5.65);
        hbkgentries[j][i] = new TH1F(Form("hbkgentries_%d_%d",j,i), "", 1, 0, 1);
        hbkgentries[j][i]->SetBinContent(1,hbkg[j][i]->GetEntries());
        hbkgentries[j][i]->SetBinError(1,hbkg[j][i]->GetBinWidth(1));
      }
    } else {
      hgen[i] = (TH1F*)feff->Get(Form("%s%s",nameeffgen.Data(),suffixSTD.Data()));
      hsel[i] = (TH1F*)feff->Get(Form("%s%s",nameeffsel.Data(),suffixSTD.Data()));

      for(int j = 0; j < nptbins; j++){
        hsig[j][i] = (TH1F*)fmass->Get(Form("%s%d_%d_1.00%s",namemasssig.Data(),ptbins[j],ptbins[j+1],suffixSTD.Data()));
        hbkg[j][i] = (TH1F*)fmass->Get(Form("%s%d_%d_1.00%s",namemassbkg.Data(),ptbins[j],ptbins[j+1],suffixSTD.Data()));
      }
    }
  }
  
  TH1F* hBkgCorr = (TH1F*)fBkgCorr->Get("hCorrFacBs");

  double fonllmin[nptbins];
  double fonllcent[nptbins];
  double fonllmax[nptbins];
  double tamucent[nptbins];

  double xmin[nptbins][trials+1];
  double xmax[nptbins][trials+1];
  double effcent[nptbins][trials+1];
  double erreffcent[nptbins][trials+1];
  
  double expectedsignalcent[nptbins][trials+1];
  double expectedsignalfonllmin[nptbins][trials+1];
  double expectedsignalfonllmax[nptbins][trials+1];
  double expectedsignalerreffmin[nptbins][trials+1];
  double expectedsignalerreffmax[nptbins][trials+1];
  
  double expectedbkg[nptbins][trials+1];
  
  double expectedsgnfcent[nptbins][trials+1];
  double expectedsgnffonllmin[nptbins][trials+1];
  double expectedsgnffonllmax[nptbins][trials+1];
  double expectedsgnferreffmin[nptbins][trials+1];
  double expectedsgnferreffmax[nptbins][trials+1];
  
  for(int j = 0; j < nptbins; j++){

    cout << filnameFONLL << endl;
    extract_fonll(filnameFONLL, j, fonllmin[j], fonllcent[j], fonllmax[j]);
    
    cout << filenameTAMU << endl;
    extract_TAMU(filenameTAMU, j, tamucent[j]);
    
    for(int i = 0; i <= trials; i++){
      cout << hsig[j][i]->GetName() << endl;
      fit_signal(hsig[j][i], j, i, xmin[j][i], xmax[j][i]);

      cout << hbkg[j][i]->GetName() << endl;
      calculate_background(hbkg[j][i], fbkg[j][i], j, i, xmin[j][i], xmax[j][i], expectedbkg[j][i]);

      cout << hgen[i]->GetName() << " " << hsel[i]->GetName() << endl;
      calculate_efficiency(hgen[i], hsel[i], j, i, effcent[j][i], erreffcent[j][i]);

      expectedsignalcent[j][i] = 2 * (ptbins[j+1] - ptbins[j]) * 1 * (BRBs * BRDs) * nEvExpected * effcent[j][i] * TAA * fonllcent[j] * tamucent[j];
      expectedsignalfonllmin[j][i] = 2 * (ptbins[j+1] - ptbins[j]) * 1 * (BRBs * BRDs) * nEvExpected * effcent[j][i] * TAA * fonllmin[j] * tamucent[j];
      expectedsignalfonllmax[j][i] = 2 * (ptbins[j+1] - ptbins[j]) * 1 * (BRBs * BRDs) * nEvExpected * effcent[j][i] * TAA * fonllmax[j] * tamucent[j];
      expectedsignalerreffmin[j][i] = 2 * (ptbins[j+1] - ptbins[j]) * 1 * (BRBs * BRDs) * nEvExpected * (effcent[j][i] - erreffcent[j][i]) * TAA * fonllcent[j] * tamucent[j];
      expectedsignalerreffmax[j][i] = 2 * (ptbins[j+1] - ptbins[j]) * 1 * (BRBs * BRDs) * nEvExpected * (effcent[j][i] + erreffcent[j][i]) * TAA * fonllcent[j] * tamucent[j];

      if(hBkgCorr->FindBin(ptbins[j]+0.01) != hBkgCorr->FindBin(ptbins[j+1]-0.01)) cout << "Warning! Different pT binning HIJING correction factor" << endl;
      expectedbkg[j][i] = hBkgCorr->GetBinContent(hBkgCorr->FindBin(ptbins[j]+0.01)) * nEvExpected * expectedbkg[j][i] / nEv335;
      
      expectedsignalcent[j][i] *= gauss3sigmafactor;
      expectedsignalfonllmin[j][i] *= gauss3sigmafactor;
      expectedsignalfonllmax[j][i] *= gauss3sigmafactor;
      expectedsignalerreffmin[j][i] *= gauss3sigmafactor;
      expectedsignalerreffmax[j][i] *= gauss3sigmafactor;

      expectedsgnfcent[j][i] = expectedsignalcent[j][i] / TMath::Sqrt(expectedsignalcent[j][i] + expectedbkg[j][i]);
      expectedsgnffonllmin[j][i] = expectedsignalfonllmin[j][i] / TMath::Sqrt(expectedsignalfonllmin[j][i] + expectedbkg[j][i]);
      expectedsgnffonllmax[j][i] = expectedsignalfonllmax[j][i] / TMath::Sqrt(expectedsignalfonllmax[j][i] + expectedbkg[j][i]);
      expectedsgnferreffmin[j][i] = expectedsignalerreffmin[j][i] / TMath::Sqrt(expectedsignalerreffmin[j][i] + expectedbkg[j][i]);
      expectedsgnferreffmax[j][i] = expectedsignalerreffmax[j][i] / TMath::Sqrt(expectedsignalerreffmax[j][i] + expectedbkg[j][i]);
    }
  }

  TFile* foutput_test = new TFile(filenameout.Data(),"RECREATE");
  for(int j = 0; j < nptbins; j++){
    cout << endl << endl;
    for(int i = 0; i <= trials; i++){
      cout << i << " " << expectedsignalcent[j][i] << " " << expectedbkg[j][i] << " " << expectedsignalcent[j][i]/expectedbkg[j][i] << " " << expectedsignalcent[j][i] / TMath::Sqrt(expectedsignalcent[j][i] + expectedbkg[j][i]) << endl;
      foutput_test->cd();
      fbkg[j][i]->Write();
      hbkgentries[j][i]->Write();
    }
  }
}

void fit_signal(TH1F* hsig, int j, int i, double &xmin, double &xmax){

  TF1* f1 = new TF1("f1", "gaus",  5.32, 5.42);
  hsig->Fit("f1", "R");
  
  double mean = f1->GetParameter(1);
  double sigma = f1->GetParameter(2);

  xmin = mean - 3 * sigma;
  xmax = mean + 3 * sigma;
}

void calculate_background(TH1F* hbkg, TF1* f1, int j, int i, double xmin, double xmax, double &bkgcent){
  hbkg->Rebin(rebin[j]);
  
  //=Chi2 fit, add "L" for likelyhood
  hbkg->Fit(Form("f_%d_%d",j,i), "R,E,+,0");
  Double_t bkgcentbc = hbkg->Integral(hbkg->FindBin(xmin),hbkg->FindBin(xmax));
  Double_t bkgcentfit = f1->Integral(xmin,xmax)/(Double_t)hbkg->GetBinWidth(1);

  if(bincountBkg[j]) bkgcent = bkgcentbc;
  else bkgcent = bkgcentfit;
}

void extract_fonll(TString filnam, int j, double &fonllmin, double &fonllcent, double &fonllmax){
  
  if(filnam=="") return 0x0;
  FILE* infil=fopen(filnam.Data(),"r");
  Char_t line[200];
  
  for(Int_t il=0; il<18; il++){
    fgets(line,200,infil);
    if(strstr(line,"central")) break;
  }
  
  Float_t ptmin,ptmax,csc,csmin,csmax,dum;
  fscanf(infil,"%f",&ptmin);
  
  Double_t relSystFF=fbtoBUnc/fbtoB;
  Double_t relSystLHCb=fLHCbBBsUnc/fLHCbBBs;
  while(!feof(infil)){
    fscanf(infil,"%f %f %f",&csc,&csmin,&csmax);
    for(Int_t i=0; i<12;i++) fscanf(infil,"%f",&dum);
    if(feof(infil)) break;
    fscanf(infil,"%f",&ptmax);
    Double_t ptmed=0.5*(ptmin+ptmax);
    Double_t dpt=(ptmax-ptmin);
    Double_t normFact=fLHCbBBs*fbtoB*1e-6/dpt; //from pb to ub/GeV/c
    csc*=normFact;
    csmin*=normFact;
    csmax*=normFact;
    Double_t systFF=relSystFF*csc;
    Double_t systLHCb=relSystLHCb*csc;
    Double_t errup=csmax-csc;
    Double_t errdw=csc-csmin;
    Double_t errtotup=TMath::Sqrt(errup*errup+systFF*systFF+systLHCb*systLHCb);
    Double_t errtotdw=TMath::Sqrt(errdw*errdw+systFF*systFF+systLHCb*systLHCb);

    ptmin=ptmax;
    
    if(ptmed > ptbins[j] && ptmed < ptbins[j+1]){
      fonllmin = csc - errtotdw;
      fonllcent = csc;
      fonllmax = csc + errtotup;
    }
  }
  fclose(infil);
}

void extract_TAMU(TString filn, int j, double &tamucent){
  
  FILE* f=fopen(filn.Data(),"r");
  Float_t pt, raa;
  Int_t iPt=0;
  Double_t meanRAA = 0.;
  Int_t ncount = 0;
  while(!feof(f)){
    fscanf(f,"%f %f\n",&pt,&raa);

    if(pt > ptbins[j] && pt <= ptbins[j+1]){
      meanRAA += raa;
      ncount++;
    }
  }
  tamucent = meanRAA / ((double)ncount);

  fclose(f);
}

void calculate_efficiency(TH1F* hgen, TH1F* hsel, int j, int i, double &effcent, double &erreffcent){

  TH1F* heff = (TH1F*)hsel->Clone("heff");
  heff->Divide(heff, hgen, 1, 1, "B");

  TH1F* heffcent = new TH1F(Form("heffcent_set%d_%d",i,j),"",1,ptbins[j],ptbins[j+1]);
  heffcent->SetBinContent(1, heff->GetBinContent(heff->FindBin(ptbins[j] + 0.5*(ptbins[j+1]-ptbins[j]))));
  heffcent->SetBinError(1, heff->GetBinError(heff->FindBin(ptbins[j] + 0.5*(ptbins[j+1]-ptbins[j]))));

  effcent = heffcent->GetBinContent(1);
  erreffcent = heffcent->GetBinError(1);
}

