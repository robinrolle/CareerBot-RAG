// components/SearchBar.js

import React, { useState, useEffect, useCallback } from 'react';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import { styled } from '@mui/material/styles';
import debounce from 'lodash.debounce';

const CustomTextField = styled(TextField)({
    '& .MuiOutlinedInput-root': {
        borderRadius: '4px', // Ajuste cette valeur pour obtenir l'arrondi désiré
    },
});

const SearchBar = ({ options, value, onChange, placeholder }) => {
    const [inputValue, setInputValue] = useState(value);

    const debouncedOnChange = useCallback(
        debounce((newValue) => {
            onChange(newValue);
        }, 300), // Adjust the debounce delay (in milliseconds) as needed
        []
    );

    useEffect(() => {
        debouncedOnChange(inputValue);
        // Cleanup function to cancel any pending debounced calls when the component unmounts
        return () => {
            debouncedOnChange.cancel();
        };
    }, [inputValue, debouncedOnChange]);

    return (
        <Autocomplete
            multiple
            options={options}
            value={value}
            onChange={(event, newValue) => {
                setInputValue(newValue);
            }}
            renderTags={() => null} // Pour masquer les Chips
            renderInput={(params) => (
                <CustomTextField
                    {...params}
                    variant="outlined"
                    placeholder={placeholder}
                />
            )}
            className="mb-4 w-full"
            disableClearable // Pour désactiver la possibilité de tout effacer
        />
    );
};

export default SearchBar;
