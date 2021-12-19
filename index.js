//On déclare une constante "tf", et on précise qu'elle nécessite le module TensforFlow.js
const tf = require('@tensorflow/tfjs');

// On déclare une variable "xs" qui contient les inputs
const xs = tf.tensor([
    [0.86226, 0.22, 0.65],  
    [0.76564, 0.24, 0.85],
    [0.53292, 0.23, 0.43],
    [0.33219, 0.26, 0.25],
    [0.38587, 0.27, 0.92],
    [0.78967, 0.21, 0.56],
    [0.63458, 0.28, 0.34],
    [0.45243, 0.25, 0.77]
]);

// On déclare une variable "ys" qui contient les outputs
const ys = tf.tensor([
  [0,1],
  [1,0],
  [0,1],
  [1,0],
  [0,1],
  [0,1],
  [1,0],
  [1,0]
]);
  // On initialise ensuite le modèle d'apprentissage
  const model = tf.sequential();

  // On ajoute une première couche cachée avec trois neurones ("units") et en précisant les dimensions des inputs ("inputShape") 
  model.add(tf.layers.dense({ units: 3, inputShape: [3], activation:'sigmoid' }));
  // On ajoute une couche de sortie qui comporte quant à elle deux neurones ("units")
  model.add(tf.layers.dense({ units: 2, activation:'sigmoid'}))
  // On compile le modèle avec des critères pour le calcul de l'erreur et l'optimisation
  model.compile({ loss: "meanSquaredError", optimizer: tf.train.adam(0.1) });

  // On passe à l'étape d'entraînement du modèle
 model.fit(xs, ys, { epochs: 5000 }).then(() => {
    // On réalise ensuite prédiction avec les données utilisées pour l'entraînement
    // model.predict(xs).print();

        /***************************************************** */

     //  model.layers[0].setWeights(model.layers[0].getWeights())    

    /* for(let i=0; i<model.getWeights().length; i++ ){

      console.log(model.getWeights()[i].dataSync())

      console.log('***************' + i +'**********************')
    }*/

     /***************************************************** */
     /*
     Tensor
           [[21.1535835 , 6.1543159  , 8.59268    ],
           [-9.6188335 , 20.0277634 , -20.411232 ],
           [-20.1396408, -26.6983356, -17.6037922]]

     console.log(model.layers[0].getWeights()[0].print()) // layer 1  // kernel:
     */

    /*

    Tensor
    [0.3941056, -6.6123557, -9.963891]

     console.log(model.layers[0].getWeights()[1].print()) // layer 1  // bias

    */  
   
  });

  