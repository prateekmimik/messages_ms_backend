import natural from "natural";
import fs from "node:fs";

// CommonJS modules can always be imported via the default export
import sqlitepkg from "sqlite3";
const { Database, OPEN_READONLY } = sqlitepkg;
import printpkg from "console-table-printer";
const { printTable } = printpkg;

const tokenizer = new natural.WordTokenizer();
const lemmatizer = natural.LancasterStemmer;
const vectorizer = new natural.TfIdf();

import tf from "@tensorflow/tfjs-node-gpu";
// import tf from "@tensorflow/tfjs-node";

class Model { }

class MultiOutputClassifierModel extends Model {

    constructor(dbName) {
        super();
        this.stopwordsEnglish = natural.stopwords;
        this.dbName = dbName;
        this.model = null;
        this.trained = false;
    }

    tableToDf(onSuccess, onError, tblName = 'MessagesCategoriesTable') {
        if (!fs.existsSync(this.dbName)) {
            onError('database file does not exist!');
            return;
        }
        try {
            const db = new Database(this.dbName, OPEN_READONLY);
            const query1 = `SELECT * FROM ${tblName} WHERE genre = 'news' LIMIT 2`;
            const query2 = `SELECT * FROM ${tblName} LIMIT 100`;
            db.all(query2,
                (err, result) => {
                    if (err) {
                        onError(err);
                    } else {
                        if (result != null && typeof result === 'object' && Array.isArray(result)) {
                            onSuccess(result);
                        } else {
                            onError('mismatching result type');
                        }
                    }
                }
            );
            db.close();
        } catch (err) {
            onError(err);
        }
    }

    encodedGenre(genre) {
        const news = genre === 'news';
        const direct = genre === 'direct';
        const social = genre === 'social';
        const result = [news, direct, social];
        if (result.every((item) => !item)) {
            // if all are false, return default as direct
            return [false, true, false];
        } else {
            return [news, direct, social];
        }
    }

    loadData(records) {
        const X = records.map(row => row.message);
        const genre = records.map(row => this.encodedGenre(row.genre));
        const ycategories = records.map(row => [
            row.aid_centers,
            row.aid_related,
            row.buildings,
            row.clothing,
            row.cold,
            row.death,
            row.direct_report,
            row.earthquake,
            row.electricity,
            row.fire,
            row.floods,
            row.food,
            row.hospitals,
            row.infrastructure_related,
            row.medical_help,
            row.medical_products,
            row.military,
            row.missing_people,
            row.money,
            row.offer,
            row.other_aid,
            row.other_infrastructure,
            row.other_weather,
            row.refugees,
            row.related,
            row.request,
            row.search_and_rescue,
            row.security,
            row.shelter,
            row.shops,
            row.storm,
            row.tools,
            row.transport,
            row.water,
            row.weather_related
        ])
        let Y = [];
        for (let i = 0; i < genre.length; i++) {
            Y.push(
                genre[i].map((item) => {
                    if (item) return 1;
                    else return 0;
                })
                .concat(ycategories[i])
            );
        }
        let categoryNames = Object.keys(records[0]);
        categoryNames.splice(0, 4);
        categoryNames.unshift('news', 'direct', 'social');
        return { X, Y, categoryNames };
    }

    tokenize(text) {
        let tokens = tokenizer.tokenize(text.toLowerCase());
        tokens = tokens.filter(token => !this.stopwordsEnglish.includes(token));
        tokens = tokens.map(token => lemmatizer.stem(token));
        return tokens;
    }

    getCombinedTfidfScores(tfidfVectorizer) {
        // Maps a proper token score matrix from tfidfVectorizer
        let uniqueTerms = new Set();
        let documentScores = [];

        // Collect all unique terms
        for (let i = 0; i < tfidfVectorizer.documents.length; i++) {
            tfidfVectorizer.listTerms(i).forEach(term => {
                uniqueTerms.add(term.term);
            });
        }
        uniqueTerms = Array.from(uniqueTerms);

        // Build the combined TF-IDF score array
        for (let i = 0; i < tfidfVectorizer.documents.length; i++) {
            let docScores = new Array(uniqueTerms.length).fill(0);
            tfidfVectorizer.listTerms(i).forEach(term => {
                let index = uniqueTerms.indexOf(term.term);
                docScores[index] = term.tfidf;
            });
            documentScores.push(docScores);
        }

        return { uniqueTerms, documentScores };
    }

    getTokensMatrix(tokenizedText) {
        // Converts a multi-dim array of tokens to tfidf matrix using vectorizer
        const documents = tokenizedText.map(doc => doc.join(" "));
        documents.forEach(doc => vectorizer.addDocument(doc));

        // Get combined TF-IDF scores
        let result = this.getCombinedTfidfScores(vectorizer);
        // printTable(result.documentScores);
        return result;
    }

    buildModel(xShape, yShape) {
        const model = tf.sequential();
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            inputShape: [xShape]
        }));
        model.add(tf.layers.dense({
            units: yShape,
            activation: 'softmax'
        }));
        model.compile({
            optimizer: 'adam', // tf.train.adam
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        this.model = model;
    }

    async trainModel(x, y) {
        const xtrain = tf.tensor2d(x);
        const ytrain = tf.tensor2d(y);

        if (!this.model) {
            console.log('model is not build');
        }
        await this.model.fit(xtrain, ytrain, {
            epochs: 50,
            batchSize: 4,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
                    this.trained = true;
                }
            }
        });
        this.saveModel();
    }

    async evaluateModel(XTest, YTest) {
        if (!this.trained) {
            console.log('model not trained');
            return;
        }
        const preds = this.model.predict(XTest);
        const accuracy = tf.metrics.binaryAccuracy(YTest, preds).mean().dataSync();
        console.log(`Model accuracy is ${accuracy}`);
    }

    async saveModel(modelFilePath = './tfjs_v1.model') {
        await this.model.save(`file://./${modelFilePath}`);
        console.log(`Model saved to ${modelFilePath}`);
    }

    async loadModel(modelFilePath = './tfjs.model') {
        const model = await tf.loadLayersModel(`${modelFilePath}/model.json`);
        return model;
    }
}

export default MultiOutputClassifierModel;
