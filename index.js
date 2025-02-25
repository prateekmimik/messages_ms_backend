import MultiOutputClassifierModel from "./lib/model.js"

const classifierModel = new MultiOutputClassifierModel('./data/dataset.db');

function app(result) {
    const { X, Y, categoryNames } = this.loadData(result);
    const tokenizedMessages = X.map(x => this.tokenize(x));
    const { uniqueTerms:tokenColumns, documentScores:tokenMatrix } = this.getTokensMatrix(tokenizedMessages);

    this.buildModel(tokenColumns.length, categoryNames.length);
    this.trainModel(tokenMatrix, Y);
}
function debugError(error) {
    console.log(error);
}
classifierModel.tableToDf(
    app.bind(classifierModel),debugError.bind(classifierModel)
);
