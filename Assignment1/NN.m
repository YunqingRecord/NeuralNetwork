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
            net=newff(minmax(obj.Inputs),obj.Labels, [10 2],{'logsig', 'purelin'}, 'trainlm');
            net.trainParam.epochs=obj.Iteration; % maximum learning step=2000
            net.trainParam.goal=0; % training goal = 0
            net.trainParam.show=1; % every single step to plot result
            net.trainParam.lr=0.005; %learning rate
            net.divideParam.valRatio = 0/100; %#ok<*PROP>
            net.divideParam.testRatio = 0/100;
            net=train(net,obj.Inputs,obj.Labels);
            obj.net = net;
        end
    end

end
