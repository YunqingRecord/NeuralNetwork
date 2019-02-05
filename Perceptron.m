classdef Perceptron % define the class Perceptron
    
    properties % define the variables involved in the class
        Inputs;
        Labels; 
        LearningRate; 
        Iteration;
        w;
        b;
        w1_revise; % used to plot the trend of change of w1
        w2_revise; % used to plot the trend of change of w2
        b_revise;  % used to plot the trend of change of bias
        loss;
        error_cal; % plot change of Total Error when training
    end
    
    methods  % define the functions used
        function obj = Perceptron() % Constructor
        end
        
        function obj = setInfo(obj,Inputs, Labels, LearningRate, Iteration)
            % core setInfo
            obj.Inputs = Inputs;
            obj.Labels = Labels;
            obj.LearningRate = LearningRate;
            obj.Iteration = Iteration;
        end
        
        function obj = Learning(obj) % Learning Algorithm
            [m, n] = size(obj.Inputs); 
            obj.w = rand(1, n)-0.5;
            obj.b = 0.8;
            obj.w1_revise = zeros(1, obj.Iteration); % Zero array to be inserted in
            obj.w2_revise = zeros(1, obj.Iteration);
            obj.b_revise  = zeros(1,  obj.Iteration);
            obj.error_cal = zeros(1,  obj.Iteration);
            for N=1:obj.Iteration
                totalerror = 0; % Clear totalerror by every iteration
                for set= 1:m
                    output = sign(obj.Inputs(set,:)*(obj.w)' + obj.b );
                    error = obj.Labels(set) - output;
                    obj.w = obj.w + obj.LearningRate * error * obj.Inputs(set);
                    obj.b = obj.b + obj.LearningRate * error;
                    totalerror = totalerror + abs(error);
                end
                obj.w1_revise(1, N) = obj.w(1);
                obj.w2_revise(1, N) = obj.w(2);
                obj.b_revise(1, N) = obj.b;
                obj.error_cal(1, N) = totalerror;
            end
        end
    end  
end