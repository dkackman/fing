const config = require('../config.js')
const path = require('path');
const { spawnSync } = require('child_process');
const { v4 } = require('uuid');

exports.generate = function (prompt) {
    sanitized = prompt
        .replace('"')
        .replace('\\')
        .trim(); // not using shell exec so don't need to owrry about shell escape characters

    const generation = config.configYaml.generation;
    const args = `run -n ${generation.environment_name} --cwd ${generation.working_dir} python ${generation.generator_py}`.split(' ');
    args.push(`"${sanitized}"`);
    const filename = v4();
    args.push(filename);

    const ps = spawnSync('conda', args);
    if (ps.status === 0) {
        return path.join(generation.output_dir, filename + '.jpg');
    }

    return '';
};
