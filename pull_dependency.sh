evaluator=evaluator.tar.gz
lib=lib.tar.gz

if [ ! -e "$evaluator" ]; then
    echo "Start downloading evaluator for overnight and geo datasets ..."
    wget -c https://worksheets.codalab.org/rest/bundles/0xbfbf0d1d8ab94874a68646a7d66c478e/contents/blob/ -O $evaluator
fi

if [ ! -e "$lib" ] ; then
    echo "Start downloading libraries for evaluation..."
    wget -c https://worksheets.codalab.org/rest/bundles/0xc6821b4f13f445d1b54e9da63019da1d/contents/blob/ -O $lib
fi

mkdir evaluator
mkdir lib
tar -zxf $evaluator -C evaluator
tar -zxf $lib -C lib
rm -rf $evaluator
rm -rf $lib
cp evaluator/sempre/module-classes.txt .

wget -c http://nlp.stanford.edu/data/glove.6B.zip
mkdir -p data/.cache
unzip glove.6B.zip -d data/.cache/
rm glove.6B.zip
