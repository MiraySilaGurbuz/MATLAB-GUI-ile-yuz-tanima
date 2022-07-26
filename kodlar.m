function varargout = matlab1(varargin)


gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @matlab1_OpeningFcn, ...
                   'gui_OutputFcn',  @matlab1_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end




function matlab1_OpeningFcn(hObject, eventdata, handles, varargin)


handles.output = hObject;


guidata(hObject, handles);




function varargout = matlab1_OutputFcn(hObject, eventdata, handles) 



varargout{1} = handles.output;




function exit_Callback(hObject, eventdata, handles)
global a
a=0;
stop(webcam);
clear handles.axes1
close all
clc
clear

% --- Executes on button press in kamera.
function kamera_Callback(hObject, eventdata, handles)
axes(handles.axes1)

veriseti = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[egitim_veriseti, test_veriseti] = splitEachLabel(veriseti, 0.7);

global net 
net = googlenet;


girdi_katman_boyutu = net.Layers(1).InputSize;

katman_grafigi = layerGraph(net);

ozellik_ogrenecek = net.Layers(142);
cikti_siniflandirici = net.Layers(144);

sinif_sayisi = numel(categories(egitim_veriseti.Labels));

yeni_ozellik_ogrenecek = fullyConnectedLayer(sinif_sayisi, ...
    'Name', 'Modelimize uygun katman', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
yeni_cikti_siniflandirici = classificationLayer('Name', 'Son katman');

katman_grafigi = replaceLayer(katman_grafigi, ozellik_ogrenecek.Name, yeni_ozellik_ogrenecek);

katman_grafigi = replaceLayer(katman_grafigi, cikti_siniflandirici.Name, yeni_cikti_siniflandirici);


Piksel_araligi = [-30 30];
olcek_araligi = [0.9 1.1];

goruntu_artirici = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', Piksel_araligi, ...
    'RandYTranslation', Piksel_araligi,... 
     'RandXScale', olcek_araligi, ...
     'RandYScale', olcek_araligi);

artirilmis_egitim_resmi = augmentedImageDatastore(girdi_katman_boyutu(1:2), egitim_veriseti, ...
    'DataAugmentation', goruntu_artirici);

artirilmis_test_resmi = augmentedImageDatastore(girdi_katman_boyutu(1:2),test_veriseti);

minibatch_boyutu = 5;
dogrulama_frekansi = floor(numel(artirilmis_egitim_resmi.Files)/minibatch_boyutu);
egitim_secenekleri = trainingOptions('sgdm',...
    'MiniBatchSize', minibatch_boyutu, ...
    'MaxEpochs', 6,...
    'InitialLearnRate', 3e-4,...
    'Shuffle', 'every-epoch', ...
    'ValidationData', artirilmis_test_resmi, ...
    'ValidationFrequency', dogrulama_frekansi, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(artirilmis_egitim_resmi, katman_grafigi, egitim_secenekleri);





kamera=webcam();
algilayici =vision.CascadeObjectDetector();


global a
a= 1;
     
    while a==1
      
    goruntu =snapshot(kamera);
    gri = rgb2gray(goruntu);
    bbox = step(algilayici,gri);
    
    
    resim = imresize(goruntu, [224, 224]);
        
    [Label, Prob] = classify(net,resim);
    isim=char(Label);
    deger=num2str(max(Prob));
    yeni_resim=insertObjectAnnotation(goruntu,"rectangle",bbox,isim+" "+deger);
    imshow(yeni_resim);
    end  


function cikma_Callback(hObject, eventdata, handles)
delete(handles.figure1);
clear



function load_Callback(hObject, eventdata, handles)
global image
[filename,pathname]= uigetfile();

if filename==0
    msgbox(sprintf('lütfen bir resim seçiniz.'),'HATA','error');
end
axes(handles.axes2)
image=imread(filename);
imshow(image);




% --- Executes on button press in pushbutton18.
function pushbutton18_Callback(hObject, eventdata, handles)
global net
global image

axes(handles.axes1)

G= imresize(image,[224, 224]);
[label, prob]= classify(net,G);
imshow(G);
title({char(label), num2str(max(prob)*100)});


