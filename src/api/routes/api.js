const express = require('express');
const generator = require('./generator.js');
var decode = require('urldecode');

const router = express.Router();

/* GET api */
router.get('/', function (req, res, next) {
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.send('This is the api root. Not much to see here.');
});

router.get('/text2img', function (req, res, next) {
  try {
    const prompt = req.query.prompt;
    res.setHeader('Content-Type', 'image/jpeg');
    //res.setHeader('Content-Length', 0);
    const filepath = generator.generate(decode(prompt));
    res.sendFile(filepath);
  }
  catch (err) {
    return res.status(500).end("an error occured");
  }
});

module.exports = router;
