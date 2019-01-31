classdef NN
    properties % define the variables involved
        Inputs;
        Labels;
        LearningRate;
        Iteration;
        w;
        b;
        w1_revise;
        w2_revise;
        b_revise;
        loss;
        error_cal;
        net
    end
    
    methods  % define the functions used
        function obj = NN()   % constructor
        end
        
        function obj = setInfo(obj, Inputs, Labels, LearningRate, Iteration)
        obj.Inputs = Inputs;
        obj.Labels = Labels;
        obj.LearningRate = LearningRate;
        obj.Iteration = Iteration;
        end
        
        function obj = Learning(obj)
            net=newff(minmax(obj.Inputs),obj.Labels, [10 1],{'logsig', 'purelin'}, 'trainlm');
            net.trainParam.epochs=2000;%允许最大训练步数2000步
            net.trainParam.goal=0; %训练目标最小误差0.001
            net.trainParam.show=1; %#ok<*PROP> %每间隔100步显示一次训练结果
            net.trainParam.lr=5; %学习速率0.05
            net.divideParam.valRatio = 0/100;
            net.divideParam.testRatio = 0/100;
            net=train(net,obj.Inputs,obj.Labels);
            obj.net = net;
        end
    end

end
