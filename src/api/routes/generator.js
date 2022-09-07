const config = require('../config.js')
const path = require('path');

exports.generate = function (prompt) {
    const filepath = path.join(config.configYaml.generation.output_dir, `${prompt}.jpg`)
    return filepath;
}