const config = require('../config.js')
const path = require('path');
const { spawnSync } = require('child_process');

exports.generate = function (prompt) {
    sanitized = prompt.replace('/[^a-z0-9áéíóúñü \.,_-]/gim,""')
        .replace('"')
        .replace('\\')
        .replace("'")
        .trim();

    const generation = config.configYaml.generation;
    const args = `run -n ${generation.environment_name} --cwd ${generation.working_dir} python ${generation.generator_py}`.split(' ');
    args.push(`"${sanitized}"`);
    const ps = spawnSync('conda', args);
    if (ps.status === 0) {
        // the resulting filename will be the first line of stdout
        const filename = String.fromCharCode(...ps.stdout).split('\n', 1)[0];

        return path.join(generation.output_dir, filename);
    }

    return '';
}
