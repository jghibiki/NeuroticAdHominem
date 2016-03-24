'use strict';

angular.module('nah.version', [
  'nah.version.interpolate-filter',
  'nah.version.version-directive'
])

.value('version', '0.1');
