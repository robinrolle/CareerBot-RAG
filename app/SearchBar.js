// components/SearchBar.js

import React from 'react';
import Select, { components } from 'react-select';

const MultiValue = ({ data }) => null;

const DropdownIndicator = () => null;

const SearchBar = ({ options, value, onChange, placeholder }) => {
    return (
        <Select
            options={options}
            isMulti
            isClearable={false}
            placeholder={placeholder}
            onChange={onChange}
            value={value}
            className="mb-4 w-full"
            components={{ MultiValue, DropdownIndicator }}
        />
    );
};

export default SearchBar;
