package com.company;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.*;

public class network {

    Vector<Vector<Float>> normalizedData= new Vector<>();
    float[][] weight_h, weight_o;

    int inputNodes, outputNodes, hiddenNodes;
    int epochs =500; //number of iterations
    float learningRate= (float) 0.01;

    String fName= "Final Weights.txt";
    DecimalFormat df = new DecimalFormat("#.####");

    network(Vector<Vector<Float>> data, int input, int hidden, int output){
        inputNodes= input;
        outputNodes= output;
        hiddenNodes= hidden;

        //update normalizedData; normalization on the input features only
        normalization(data);

        //Initialize the weights (parameters) randomly.
        weight_h= new float[hiddenNodes][inputNodes+1];//plus one to add x0
        weight_o= new float[outputNodes][hiddenNodes];
    }


//****************************************************The First Program*************************************************

    public void Run() throws IOException {
        double[] MSE = new double[normalizedData.size()]; //for each training example

        //Initialize weight_h matrix
        for (int i=0; i<hiddenNodes; i++){
            for (int j=0; j<=inputNodes ; j++){ weight_h[i][j]= getRandomNumber(); }
        }
        //Initialize weight_O matrix
        for (int i=0; i<outputNodes; i++){
            for (int j=0; j<hiddenNodes ; j++){ weight_o[i][j]= getRandomNumber(); }
        }

        System.out.println("First Program is running:");

        //1. Loop over epochs:
        for(int itr=0; itr< epochs; itr++){

            //2. Loop over each training example/row:
            for(int i=0; i<normalizedData.size(); i++){ //one layer
                float[] inputLayer  = new float[inputNodes+1], //plus one to add x0
                        outputFromHiddenLayer = new float[hiddenNodes],
                        outputLayer = new float[outputNodes],
                        outputFromOutputLayer = new float[outputNodes];


                //add  values to inputLayer and outputLayer
                inputLayer[0]=1; //add x0 = 1
                for(int j=0; j< normalizedData.get(i).size(); j++) {
                    if (j < inputNodes) {
                        inputLayer[j+1] = normalizedData.get(i).get(j);
                    } else {
                        outputLayer[j-inputNodes] = normalizedData.get(i).get(j);
                    }
                }

            //-------------------------------FeedForward----------------------------------------------------------------

                //3. add values to outputFromHiddenLayer => input to output layer
                outputFromHiddenLayer= FeedForward(weight_h, inputLayer);

                //4. add values to outputFromOutputLayer
                outputFromOutputLayer= FeedForward(weight_o, outputFromHiddenLayer);

            //-------------------------------BackPropagation------------------------------------------------------------

                float[] deltaErrorOfOutputLayer= new float[outputNodes],
                        deltaErrorOfHiddenLayer= new float[hiddenNodes];

                //5. calculate delta For each output neuron k
                for (int k=0; k<outputNodes; k++){
                    // 𝛅ok = (𝒂k – 𝒚k) ∗ 𝒂ok ∗ (𝟏 − 𝒂ok)
                    deltaErrorOfOutputLayer[k]= Float.parseFloat(df.format((outputFromOutputLayer[k]-outputLayer[k]) * outputFromOutputLayer[k] * (1-outputFromOutputLayer[k])));
                }

                //6. calculate delta For each hidden neuron j
                for (int j=0; j<hiddenNodes; j++){
                    // 𝛅hj = ( ∑ 𝛅o𝒌 ∗ 𝒘o𝒌𝒋 ) ∗ 𝒂h𝒋 ∗ (𝟏 − 𝒂h𝒋)
                    float sum = 0;
                    for (int k=0; k<outputNodes; k++){
                        sum+=  (deltaErrorOfOutputLayer[k] * weight_o[k][j] );
                    }

                    deltaErrorOfHiddenLayer[j]= Float.parseFloat(df.format(sum * outputFromHiddenLayer[j] * (1-outputFromHiddenLayer[j])));
                }

            //-------------------------------WeightUpdate---------------------------------------------------------------

                //7. For each weight 𝒘o𝒌𝒋 going to the output layer:
                 //Update 𝒘o𝒌𝒋 = 𝒘o𝒌𝒋 – η * 𝛅o𝒌 * 𝒂h𝒋
                for (int k=0; k<weight_o.length; k++){
                    for (int j=0; j<weight_o[k].length; j++){
                        weight_o[k][j] -= Float.parseFloat(df.format((learningRate * deltaErrorOfOutputLayer[k] * outputFromHiddenLayer[j])));
                    }
                }

                //8. For each weight 𝒘h𝒋i going to the hidden layer:
                //Update 𝒘h𝒋l = 𝒘h𝒋l – η * 𝛅hj * xl
                for (int j=0; j<weight_h.length; j++) {
                    for (int l = 0; l < weight_h[j].length; l++) {
                        weight_h[j][l] -= Float.parseFloat(df.format((learningRate * deltaErrorOfHiddenLayer[j] * inputLayer[l])));
                    }
                }

            //-----------------------------------Calculate the mean square error----------------------------------------

                //9. calculate MSE; if it is the last iteration
                if (itr+1==epochs){
                    float sumOfError=0;
                    for (int k=0;k <outputNodes;k++){ // error foreach neuron in the output Layer
                        sumOfError += (outputFromOutputLayer[k]-outputLayer[k]);
                    }
                    MSE[i]= Double.parseDouble(df.format(0.5* Math.pow(sumOfError,2))); //MSE/overall error at the output layer
                }

            } //end of training data size

        } //end of epochs

        //output:
        //1. print that MSE on the screen
        float sum=0;
        for(int i=0; i< MSE.length; i++){
            sum+= MSE[i];
        }
        System.out.println("    Mean MSE for Training Data = "+ df.format(sum/normalizedData.size()));

        // 2. save your final weights to a file
        writeToFile(weight_h, "Hidden Layer");
        writeToFile(weight_o, "Output Layer");

        System.out.println("    Final Weights saved to a file\n");

    }

//****************************************************The Second Program************************************************

    public void test(Vector<Vector<Float>> testData){
        float[] MSE= new float[testData.size()];

        System.out.println("Second Program is running:");

        for(int i=0; i<testData.size(); i++) { //one layer
            float[] inputLayer = new float[inputNodes + 1], //plus one to add x0
                    outputFromHiddenLayer = new float[hiddenNodes],
                    outputLayer = new float[outputNodes],
                    outputFromOutputLayer = new float[outputNodes];


            //add  values to inputLayer and outputLayer
            inputLayer[0] = 1; //add x0 = 1
            for (int j = 0; j < testData.get(i).size(); j++) {
                if (j < inputNodes) {
                    inputLayer[j + 1] = testData.get(i).get(j);
                } else {
                    outputLayer[j - inputNodes] = testData.get(i).get(j);
                }
            }

            //-------------------------------FeedForward----------------------------------------------------------------

            //1. add values to outputFromHiddenLayer => input to output layer
            outputFromHiddenLayer = FeedForward(weight_h, inputLayer);

            //2. add values to outputFromOutputLayer
            outputFromOutputLayer = FeedForward(weight_o, outputFromHiddenLayer);

            //-----------------------------------Calculate the mean square error----------------------------------------

            //3. calculate MSE;
            float sumOfError=0;
            for (int k=0;k <outputNodes;k++){ // error foreach neuron in the output Layer
                sumOfError += (outputFromOutputLayer[k]-outputLayer[k]);
            }
            MSE[i]= Float.parseFloat(df.format(0.5* Math.pow(sumOfError,2))); //MSE/overall error at the output layer
        }

        //output:
        //1. print that MSE on the screen
        float sum=0;
        for(int i=0; i< MSE.length; i++){
            sum+= MSE[i];
        }
        System.out.println("    Mean MSE for Test Data = "+ df.format(sum/normalizedData.size()));

    }


//------------------------------------------------Sub Functions---------------------------------------------------------

    private void normalization(Vector<Vector<Float>> data){
        //step 1: calculate mean for each column
        Vector<Float> columnMEan= new Vector(); //size = inputNodes + outputNodes

        //size-outputNodes => to skip output feature
        for (int i=0; i<data.get(0).size(); i++){ //data.get(0).size= inputNodes + outputNodes, loop on columns
            float sumOfcol=0;
            for(int j=0; j<data.size(); j++){ //loop on rows
                sumOfcol+= data.get(j).get(i);
            }
            columnMEan.add(sumOfcol/data.size());
        }

        //step 2: calculate std dev
        Vector<Float> stdDev= new Vector<>();
        for (int i=0; i<data.get(0).size(); i++){
            float sumOfcol=0;
            for(int j=0; j<data.size(); j++){
                sumOfcol+= Math.pow((data.get(j).get(i) - columnMEan.get(i)),2);
            }
            stdDev.add((float) Math.sqrt(sumOfcol/data.size()));
        }

        //step 3: add to normalized data; v' = (v - mean) / std dev
        for (int i=0; i<data.size(); i++){
            Vector<Float> row= new Vector<>();
            for(int j=0; j<data.get(i).size()-outputNodes; j++) {
                float newValue = (data.get(i).get(j) - columnMEan.get(j)) / stdDev.get(j);
                row.add(newValue);
            }
            /*
            //add value of feature output
            for (int o=0; o<outputNodes; o++) {
                row.add(data.get(i).get(inputNodes+o));
            }
             */
            normalizedData.add(row);
        }
    }

    private float getRandomNumber() {
        int max=1, min=-1;
        return (float) Math.random() * (max - min) + min;
    }

    private float[] FeedForward(float[][]weight, float[] input){
        float[] output= new float[weight.length];

        for (int i=0; i<weight.length; i++){
            float sum=0;
            for (int j=0; j<weight[i].length; j++){
                sum+= Float.parseFloat(df.format(weight[i][j] * input[j]));
            }

            //System.out.println(sum);

            output[i]=  Float.parseFloat(df.format((1/(1 + Math.exp(-1*sum)))));
        }

        return output;
    }

    private void writeToFile(float[][] mat, String matName) throws IOException {
        saveToFile("Weight matrix for "+matName+": ["+mat[0].length+"x"+mat.length+"]\n");
        for(int i=0; i<mat.length; i++){
            String line= "|";
            for (int j=0; j< mat[i].length; j++){
                String str=df.format(mat[i][j]);
                while (str.length()<10){
                    str+=" ";
                }
                line+= (str);
            }
            line+="|\n";
            saveToFile(line);
        }
        saveToFile("--------------------------------------------------------------------------------------\n");
    }

    private void saveToFile(String data) throws IOException {
            try{
                File file= new File(fName);

                //not exist->create new file
                if(!(file.exists())){
                    //System.out.println("create new file");
                    Path filepath = file.toPath(); //convert String to Path
                    Files.createFile(filepath);
                }
                //file exist
                FileWriter fr = new FileWriter(file, true); //true -> to append
                fr.write(data+"\n");
                fr.close();

            }catch (IOException e) {
                System.out.println("error!");
            }
    }

}
