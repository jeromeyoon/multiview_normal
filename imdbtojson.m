clear all
close all

addpath('/home/yjyoon/work/eccv_matlab/jsonlab')
savepath = '/research2/PAMI_2018/dataset/';
load('eccv_journal2.mat');

trainidx = find(imdb.sets ==1);
testdx = find(imdb.sets ==2);


% traininput = cell2mat(imdb.dt(trainidx));
% traingt = cell2mat(imdb.gt(trainidx));
% testinput = cell2mat(imdb.dt(testdx));
% testgt = cell2mat(imdb.gt(testdx));

traininput1 =imdb.dt(trainidx);
%traininput2 =imdb.high_dt(trainidx);
%traininput2 =imdb.light(trainidx);
traingt1 =imdb.gt(trainidx);
%traingt2 =imdb.high_gt(trainidx);
%traingt3 =imdb.low2_gt(trainidx);

testinput1 =imdb.dt(testdx);
%testinput2 =imdb.low_dt(testdx);
%testinput3 =imdb.low2_dt(testdx);
%testinput2 = imdb.light(testdx);
testgt1 = imdb.gt(testdx);
%testgt2 = imdb.low_gt(testdx);
%testgt3 = imdb.low2_gt(testdx);


%assert(length(traininput1)==length(traininput2));
assert(length(traininput1)==length(traingt1));

%assert(length(testinput1)==length(testinput2));
%assert(length(testinput1)==length(testgt1));


%traininput = cell2mat(traininput1);
%hightraininput = cell2mat(traininput2);
%trainlow2 = cell2mat(traininput3);
%traingt = cell2mat(traingt1);
%hightraingt = cell2mat(traingt2);
%traingtlow = cell2mat(traingt2);
%traingtlow2 = cell2mat(traingt3);

%testinput = cell2mat(testinput1);
%testlow = cell2mat(testinput2);
%testlow2 = cell2mat(testinput3);
%testgt = cell2mat(testgt1);
%testlow = cell2mat(testgt2);
%testlow2 = cell2mat(testgt3);
% num_traindata = length(traininput1);
% num_testinput = length(testinput1);

% step = 999;
% tmp_traininput=[];
% tmp_traingt=[];
% tmp_testnput=[];
% tmp_testgt=[];
% count =1;
% 
% for i=1:step:num_traindata
%     
%     t = step*count;
%      fprintf('%d/%d \n',i,t );
%     tmp = cell2mat(traininput(i:t));
%     tmp_traininput =cat(1,tmp_traininput,tmp);
%     
%     tmp2 = cell2mat(traingt(i:t));
%     tmp_traingt =cat(1,tmp_traingt,tmp2);
%     count = count+1;
% end
% traininput = tmp_trainput;
% traingt = tmp_traingt;
% 
% count =1;
% for i=1:step:num_testinput
%      t = step*count;
%     tmp = cell2mat(testinput(i:t));
%     tmp_testinput =cat(1,tmp_testinput,tmp);
%     
%     tmp2 = cell2mat(testgt(i:t));
%     tmp_testgt =cat(1,tmp_testgt,tmp2);
%     count = count+1;
% end
% testinput = tmp_testinput;
% testgt = tmp_testgt;


savejson('',traininput1,fullfile(savepath,'traininput_5678.json'));
%savejson('',traininput2,fullfile(savepath,'high_traininput.json'));

savejson('',traingt1,fullfile(savepath,'traingt_5678.json'));
%savejson('',traingt2,fullfile(savepath,'high_traingt.json'));
savejson('',testinput1,fullfile(savepath,'testinput_5678.json'));
%savejson('',testlow,fullfile(savepath,'testlow.json'));
%savejson('',testlow2,fullfile(savepath,'testlow2.json'));
%savejson('',testlight,fullfile(savepath,'testlight.json'));
savejson('',testgt1,fullfile(savepath,'testgt_5678.json'));

%load json file 
%loadjson('testinput_material.json')