function [Data_SubPredict, Data_SubTrain] = Data_Rate(Data_Train, TestRate)

% Positive data
  PosI = Data_Train(:, end)==1;
  PosData_Train = Data_Train(PosI, :);
  PosM_Train = sum(PosI);
  
% Positive data for SubPredict
  PosI_SubPredict = sort(randperm(PosM_Train, round(TestRate*PosM_Train)));
  PosSamples_SubPredict = PosData_Train(PosI_SubPredict, 1:end-1);
  PosLabels_SubPredict = PosData_Train(PosI_SubPredict, end);
  
% Positive data for SubTrain
  PosI_SubTrain = setdiff(1:PosM_Train, PosI_SubPredict);
  PosSamples_SubTrain = PosData_Train(PosI_SubTrain, 1:end-1);
  PosLabels_SubTrain = PosData_Train(PosI_SubTrain, end);
  
% Negative data
  NegI = Data_Train(:, end)==-1;
  NegData_Train = Data_Train(NegI, :);
  NegM_Train = sum(NegI);
  
% Negative data for SubPredict
  NegI_SubPredict = sort(randperm(NegM_Train, round(TestRate*NegM_Train)));
  NegSamples_SubPredict = NegData_Train(NegI_SubPredict, 1:end-1);
  NegLabels_SubPredict = NegData_Train(NegI_SubPredict, end);
  
% Negative data for SubTrain
  NegI_SubTrain = setdiff(1:NegM_Train, NegI_SubPredict);
  NegSamples_SubTrain = NegData_Train(NegI_SubTrain, 1:end-1);
  NegLabels_SubTrain = NegData_Train(NegI_SubTrain, end);
  
% SubPredict data
  Samples_SubPredict = [PosSamples_SubPredict; NegSamples_SubPredict];
  Labels_SubPredict = [PosLabels_SubPredict; NegLabels_SubPredict];
  Data_SubPredict = [Samples_SubPredict, Labels_SubPredict];
  M_SubPredict = size(Data_SubPredict, 1);
  Data_SubPredict = Data_SubPredict(randperm(M_SubPredict), :);
  
  
% SubTrain data
  Samples_SubTrain = [PosSamples_SubTrain; NegSamples_SubTrain];
  Labels_SubTrain = [PosLabels_SubTrain; NegLabels_SubTrain];
  Data_SubTrain = [Samples_SubTrain, Labels_SubTrain];
  M_SubTrain = size(Data_SubTrain, 1);
  Data_SubTrain = Data_SubTrain(randperm(M_SubTrain), :);

end

